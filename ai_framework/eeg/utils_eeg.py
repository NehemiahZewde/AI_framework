# utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable, MutableMapping, Hashable
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import h5py
import mne
import json
from collections import Counter, defaultdict
from copy import deepcopy
import re
import pickle

from .eeg_preprocess import eeg_preprocess_pipeline, plot_pipeline_text, config_nehemiah
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from scipy.io import loadmat

from collections.abc import Sequence
from time import perf_counter

AggName = Literal["mean", "median","nanmean", "nanmedian", "sum", "min", "max"]
AggFn = Callable[[NDArray[np.floating], int], NDArray[np.floating]]  # not used directly, see note below
Agg = Union[AggName, Callable[..., Any]]  # accept np.mean / np.median etc.




def aggregate_bundle_by_group(
    bundle: Mapping[str, Any],
    agg: Agg = "mean",
    *,
    x_key: str = "X_raw",
    out_prefix: str = "combined_",
    keep_original: bool = True,
) -> Dict[str, Any]:
    """
    Aggregate a sample-level dataset ("bundle") into a group-level dataset by grouping rows
    using `bundle["groups"]` and applying an aggregation function (default: mean) to the
    feature rows in `bundle[x_key]`.

    This is designed for the common setting where you have multiple samples per subject
    (or per recording/session), and you want exactly one row per group.

    Expected input structure (minimum required keys):
        - bundle[x_key]: 2D numpy array of shape (N, D) with per-sample features
        - bundle["groups"]: 1D numpy int array of shape (N,) with group ids per sample

    Optional metadata keys (recommended):
        - bundle["group_id_to_key"]: dict[int, (label_str, subject_id)] mapping each group id
          to a tuple like ("ASD", "NDAR...").
        - bundle["label_to_id"]: dict[str, int] mapping class name (e.g., "ASD") to numeric id

    Label aggregation:
        - If both `group_id_to_key` and `label_to_id` exist, group labels are constructed
          from them (preferred; deterministic and consistent with your metadata).
        - Otherwise, it falls back to using sample-level `bundle["y"]` and takes either:
            * the unique label if all samples in a group match, OR
            * majority vote if mixed labels are present.

    Assumptions:
        - Group ids are contiguous integers starting at 0 with no gaps:
              groups ∈ {0, 1, ..., G-1}
          If this is violated (e.g., missing group id), a ValueError is raised.

    Parameters
    ----------
    bundle:
        Input mapping containing arrays and metadata (your dataset dictionary).
    agg:
        Aggregation to apply within each group. You may pass:
          - strings: {"mean","median","nanmean","nanmedian","sum","min","max"} OR
          - callable: e.g., np.mean, np.nanmean, np.median, np.nanmedian (must accept axis=0)
    x_key:
        Key in `bundle` that points to the feature matrix to aggregate (default: "X_raw").
    out_prefix:
        Prefix for new keys when `keep_original=True`. Default "combined_".
        Example outputs: "combined_X_raw", "combined_y", ...
    keep_original:
        If True (default), preserve original sample-level arrays and *add* aggregated arrays
        under new keys.
        If False, overwrite `x_key`, "y", and "groups" with aggregated versions.

    Returns
    -------
    Dict[str, Any]
        A deep-copied, updated dictionary containing aggregated arrays.

        If keep_original=True (default), new keys added include:
            - f"{out_prefix}{x_key}": aggregated features, shape (G, D)
            - f"{out_prefix}y": aggregated labels, shape (G,)
            - f"{out_prefix}groups": group ids, shape (G,) typically [0..G-1]
            - f"{out_prefix}n_per_group": sample counts per group, shape (G,)
            - f"{out_prefix}agg": name of aggregation used

        If keep_original=False, the above are stored as:
            - bundle[x_key], bundle["y"], bundle["groups"], plus "n_per_group" and "agg"

    Raises
    ------
    KeyError:
        If required keys (x_key or "groups") are missing.
    ValueError:
        If shapes are inconsistent, `bundle[x_key]` is not 2D, or group ids are not contiguous.

    Examples
    --------
    >>> bundle2 = aggregate_bundle_by_group(bundle, agg="mean")
    >>> Xg = bundle2["combined_X_raw"]  # (G, D)
    >>> yg = bundle2["combined_y"]      # (G,)
    >>> counts = bundle2["combined_n_per_group"]

    >>> # Overwrite in-place style (but returned as a new dict)
    >>> bundle_group = aggregate_bundle_by_group(bundle, agg=np.median, keep_original=False)
    >>> bundle_group["X_raw"].shape
    (G, D)
    """
    # --------- Extract core arrays ---------
    if x_key not in bundle:
        raise KeyError(f"Missing key '{x_key}' in bundle.")
    if "groups" not in bundle:
        raise KeyError("Missing key 'groups' in bundle.")

    X: NDArray[np.floating] = np.asarray(bundle[x_key])
    groups: NDArray[np.integer] = np.asarray(bundle["groups"])

    # --------- Validate shapes ---------
    if X.ndim != 2:
        raise ValueError(f"{x_key} must be a 2D array of shape (N, D). Got shape {X.shape}.")
    if groups.ndim != 1:
        raise ValueError(f"'groups' must be a 1D array of shape (N,). Got shape {groups.shape}.")
    if groups.shape[0] != X.shape[0]:
        raise ValueError(
            f"Length mismatch: {x_key} has N={X.shape[0]} rows but groups has length {groups.shape[0]}."
        )

    # --------- Infer number of groups (contiguous 0..G-1) ---------
    # Because ids are contiguous with no gaps, we can safely use max+1.
    g_min = int(groups.min())
    g_max = int(groups.max())
    if g_min != 0:
        raise ValueError(f"Expected group ids to start at 0. Found min(groups)={g_min}.")

    G = g_max + 1  # total groups

    # Compute samples-per-group and verify no gaps (counts==0 indicates missing group id)
    counts: NDArray[np.int64] = np.bincount(groups.astype(np.int64), minlength=G)
    if np.any(counts == 0):
        # This contradicts the stated assumption "no gaps", so fail loudly.
        missing = np.where(counts == 0)[0]
        raise ValueError(f"Found gaps: these group ids have zero samples: {missing[:20]}{'...' if missing.size>20 else ''}")

    # --------- Resolve aggregation function ---------
    # We accept either a known string or a callable like np.mean/np.median that supports axis=0.
    if isinstance(agg, str):
        agg_map: Dict[str, Callable[..., Any]] = {
            "mean": np.mean,
            "median": np.median,
            "nanmean": np.nanmean,       
            "nanmedian": np.nanmedian,   
            "sum": np.sum,
            "min": np.min,
            "max": np.max,
        }
        if agg not in agg_map:
            raise ValueError(f"Unknown agg='{agg}'. Use {list(agg_map.keys())} or pass a callable.")
        agg_fn = agg_map[agg]
        agg_name = agg
    elif callable(agg):
        agg_fn = agg
        agg_name = getattr(agg, "__name__", "custom")
    else:
        raise ValueError("agg must be a string or a callable (e.g., np.mean).")

    # --------- Aggregate X per group ---------
    # Pre-allocate output as float32 (matching your X_raw dtype) unless you prefer float64.
    Xg: NDArray[np.float32] = np.empty((G, X.shape[1]), dtype=np.float32)

    # Main loop: for each group, select its rows and reduce along axis=0 to get one D-dim vector
    for g in range(G):
        mask = (groups == g)  # boolean mask for samples in group g
        X_block = X[mask]     # shape (n_g, D)

        # Apply aggregation across the rows within this group => shape (D,)
        # Note: agg_fn must support axis=0 (np.mean/np.median/... do).
        Xg[g] = agg_fn(X_block, axis=0)

    # --------- Aggregate y per group ---------
    group_id_to_key: Optional[Mapping[int, Tuple[str, str]]] = bundle.get("group_id_to_key")
    label_to_id: Optional[Mapping[str, int]] = bundle.get("label_to_id")

    if group_id_to_key is not None and label_to_id is not None:
        # Preferred: use metadata mapping group -> label_str -> label_id
        yg: NDArray[np.int32] = np.empty((G,), dtype=np.int32)
        for g in range(G):
            label_str, _subject_id = group_id_to_key[g]
            try:
                yg[g] = int(label_to_id[label_str])
            except KeyError as e:
                raise KeyError(f"label_to_id missing label '{label_str}' for group {g}.") from e
    else:
        # Fallback: derive group label from sample-level y via uniqueness / majority vote
        if "y" not in bundle:
            raise KeyError("Missing key 'y' in bundle and cannot infer group labels without metadata.")

        y: NDArray[np.integer] = np.asarray(bundle["y"])
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Length mismatch: y has length {y.shape[0]} but X has {X.shape[0]} rows.")

        yg = np.empty((G,), dtype=np.int32)
        for g in range(G):
            vals = y[groups == g]
            uniq = np.unique(vals)

            if uniq.size == 1:
                # Clean case: all samples in the group share the same class id
                yg[g] = int(uniq[0])
            else:
                # Mixed-label group: choose majority class id
                # NOTE: This indicates potential data issues; you may prefer to raise instead.
                yg[g] = int(np.bincount(vals.astype(np.int64)).argmax())

    # After aggregation, each row corresponds 1:1 to a group id 0..G-1
    group_ids: NDArray[np.int32] = np.arange(G, dtype=np.int32)

    # --------- Build returned bundle (deep copy to avoid mutating original) ---------
    new_bundle: Dict[str, Any] = deepcopy(dict(bundle))

    # Store results either under new keys or overwrite existing keys
    if keep_original:
        new_bundle[f"{out_prefix}{x_key}"] = Xg
        new_bundle[f"{out_prefix}y"] = yg
        new_bundle[f"{out_prefix}groups"] = group_ids
        new_bundle[f"{out_prefix}n_per_group"] = counts.astype(np.int32)
        new_bundle[f"{out_prefix}agg"] = agg_name
    else:
        new_bundle[x_key] = Xg
        new_bundle["y"] = yg
        new_bundle["groups"] = group_ids
        new_bundle["n_per_group"] = counts.astype(np.int32)
        new_bundle["agg"] = agg_name

    return new_bundle



def add_recording_date_order_from_filename(
    df: pd.DataFrame,
    *,
    filename_col: str = "filename",
    subject_col: str = "subject_id",
    date_regex: str = r"^[^_]+_(\d{8})_r\.mat$",
    date_format: str = "%Y%m%d",
    date_col: str = "recording_date",
    order_col: str = "recording_order",
    n_recordings_col: str = "n_recordings_for_subject",
    keep_date_str: bool = False,
) -> pd.DataFrame:
    """
    Add recording date and within-subject recording order to a scanned EEG dataframe.

    This is intended for datasets where recording date is encoded in the filename,
    such as ABCCT files:

        NDARAA898JB2_20180207_r.mat

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by scan_eeg_directory() or equivalent.

    filename_col : str
        Column containing filenames.

    subject_col : str
        Column containing subject IDs.

    date_regex : str
        Regex with exactly one capture group for the date string.
        Default captures YYYYMMDD from ABCCT filenames.

    date_format : str
        Datetime format used by pd.to_datetime.

    date_col : str
        Name of the parsed datetime column.

    order_col : str
        Name of the within-subject chronological recording order column.

    n_recordings_col : str
        Name of the column storing total recordings per subject.

    keep_date_str : bool
        If True, keeps the intermediate raw date string column.

    Returns
    -------
    pd.DataFrame
        Copy of df with recording_date, recording_order, and
        n_recordings_for_subject inserted after subject_id.
    """

    required = {filename_col, subject_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")

    out = df.copy()

    date_str_col = f"{date_col}_str"

    out[date_str_col] = out[filename_col].astype(str).str.extract(date_regex)[0]
    out[date_col] = pd.to_datetime(
        out[date_str_col],
        format=date_format,
        errors="coerce",
    )

    # Stable chronological ordering within subject.
    # Files with missing/unparseable dates are placed last.
    out = (
        out.sort_values(
            [subject_col, date_col, filename_col],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    out[order_col] = out.groupby(subject_col).cumcount() + 1
    out[n_recordings_col] = out.groupby(subject_col)[filename_col].transform("count")

    if not keep_date_str:
        out = out.drop(columns=[date_str_col])

    # Move new columns immediately after subject_col
    new_cols = [date_col, order_col, n_recordings_col]
    cols = list(out.columns)

    for c in new_cols:
        cols.remove(c)

    insert_at = cols.index(subject_col) + 1
    cols = cols[:insert_at] + new_cols + cols[insert_at:]

    return out[cols]

def plot_total_and_value_counts(
    df: pd.DataFrame,
    *,
    col: str,
    figsize=(12, 5),
    font_size: int = 12,
    title: str | None = None,
    xlabel: str = "Value",
    ylabel: str = "Count",
    total_bar_label: str = "Total",
    missing_label: str = "Missing",
    show_percent: bool = True,
    show_total: bool = True,
    rotate_xticks: int = 0,
    class_colors: dict | None = None,   # keys must match bars shown (Total + all value labels)
    sns_style: str = "whitegrid",
):
    """
    Generic, presentation-ready distribution plot from ONE column.

    Bars:
      - Total bar = len(df)
      - Category bars = df[col].value_counts(dropna=False)

    Percentages:
      - Uses value_counts(normalize=True) * 100, including NaN

    Missing:
      - NaN is displayed as `missing_label`

    Styling:
      - Seaborn barplot (with hue='Label' to avoid seaborn palette deprecation warning)
      - Bold fonts, tight layout
      - Annotates each bar with N and (%) of total
    """

    if col not in df.columns:
        raise KeyError(f"col '{col}' not found. Available columns: {list(df.columns)}")

    sns.set_style(sns_style)

    s = df[col]

    # counts + percents (including NaN)
    counts = s.value_counts(dropna=False)
    percents = s.value_counts(dropna=False, normalize=True) * 100.0

    def pretty(v):
        return missing_label if pd.isna(v) else str(v)

    # Pretty labels for indices (preserve order from value_counts)
    counts.index = [pretty(v) for v in counts.index]
    percents.index = [pretty(v) for v in percents.index]

    total = int(len(df))

    # Prepend total bar
    plot_counts = pd.concat([pd.Series({total_bar_label: total}), counts])
    plot_percents = pd.concat([pd.Series({total_bar_label: 100.0}), percents])

    plot_df = pd.DataFrame({
        "Label": plot_counts.index,
        "Count": plot_counts.values,
        "Percent": [float(plot_percents.loc[k]) for k in plot_counts.index],
    })

    # preserve order: Total first, then value_counts order
    order = plot_df["Label"].tolist()

    palette = None
    if class_colors is not None:
        if not isinstance(class_colors, dict):
            raise TypeError("class_colors must be a dict mapping bar labels to color strings.")
        missing_keys = [k for k in order if k not in class_colors]
        if missing_keys:
            raise ValueError(
                f"class_colors is missing colors for: {missing_keys}. "
                f"Provide colors for all of: {order}"
            )
        palette = {k: class_colors[k] for k in order}

    fig, ax = plt.subplots(figsize=figsize)

    # seaborn >=0.13: palette without hue deprecated → use hue='Label' and legend=False
    sns.barplot(
        data=plot_df,
        x="Label",
        y="Count",
        order=order,
        hue="Label",
        hue_order=order,
        palette=palette,
        dodge=False,
        legend=False,
        ax=ax,
    )

    ax.set_title(title or f"Distribution of {col}", fontsize=font_size + 2, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right" if rotate_xticks else "center")

    # annotations
    y_max = float(plot_df["Count"].max()) if len(plot_df) else 0.0
    pad = max(1.0, 0.02 * y_max) if y_max > 0 else 1.0

    for p, (_, row) in zip(ax.patches, plot_df.iterrows()):
        v = float(row["Count"])
        if v <= 0:
            continue
        pct = float(row["Percent"])

        lines = [f"N={int(v)}"] if show_total else []
        if show_percent:
            lines.append(f"({pct:.1f}%)")

        ax.text(
            p.get_x() + p.get_width() / 2,
            v + pad,
            "\n".join(lines),
            ha="center",
            va="bottom",
            fontsize=font_size - 1,
            fontweight="bold",
        )

    ax.set_ylim(0, (y_max * 1.18 + 1) if y_max > 0 else 1)
    plt.tight_layout()
    plt.show()


def plot_histogram(
    df: pd.DataFrame,
    *,
    col: str,
    bins: int | str = "auto",          # int, "auto", "fd", "sturges", etc.
    figsize=(12, 5),
    font_size: int = 12,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Count",
    sns_style: str = "whitegrid",
    kde: bool = False,
    show_missing_bar: bool = True,
    missing_label: str = "Missing",
    missing_color: str = "gray",
    annotate: bool = True,
):
    """
    Presentation-ready histogram for ONE numeric column.
    - Plots histogram of non-missing numeric values
    - Optionally adds a separate bar showing the count of missing values
    - Optional KDE overlay
    """

    if col not in df.columns:
        raise KeyError(f"col '{col}' not found. Available columns: {list(df.columns)}")

    sns.set_style(sns_style)

    # Coerce to numeric (non-numeric -> NaN)
    s = pd.to_numeric(df[col], errors="coerce")
    n_total = len(s)
    n_missing = int(s.isna().sum())
    s_nonmissing = s.dropna()

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram (non-missing)
    sns.histplot(
        s_nonmissing,
        bins=bins,
        kde=kde,
        ax=ax,
    )

    ax.set_title(title or f"Histogram of {col}", fontsize=font_size + 2, fontweight="bold")
    ax.set_xlabel(xlabel or col, fontsize=font_size, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")

    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")

    # Optional missing bar on a twin axis (so it doesn't distort histogram scale)
    if show_missing_bar and n_missing > 0:
        ax2 = ax.twinx()
        ax2.bar([0], [n_missing], color=missing_color, alpha=0.35, width=0.6)
        ax2.set_ylabel(f"{missing_label} count", fontsize=font_size, fontweight="bold")
        ax2.tick_params(axis="y", labelsize=font_size)
        for t in ax2.get_yticklabels():
            t.set_fontweight("bold")
        ax2.set_xticks([])

        if annotate:
            ax2.text(
                0,
                n_missing,
                f"{missing_label}: N={n_missing} ({(n_missing/n_total)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=font_size - 1,
                fontweight="bold",
            )

    # Annotate overall non-missing N
    if annotate:
        ax.text(
            0.99,
            0.98,
            f"Non-missing N={len(s_nonmissing)} ({(len(s_nonmissing)/n_total)*100:.1f}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=font_size - 1,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.show()


def find_files_with_hints(
    in_dir: str | Path,
    folder_hints: List[str],
    file_pattern: str,
) -> List[Path]:
    """
    Recursively search a directory using a sequence of folder hints that act as
    successive filters, eventually returning files that match a given pattern.

    This function is designed for deeply nested directory structures (e.g., BIDS)
    where users want to specify a hierarchical search such as:
        ["sub-*", "ses-1", "eeg"]

    Meaning:
    --------
    - Start in the root directory `in_dir`
    - Find all subdirectories matching "sub-*"
    - Inside those, find subdirectories matching "ses-1"
    - Inside those, find subdirectories matching "eeg"
    - Then, within the final matched directories, search for files that match
      `file_pattern` (e.g., "*_eeg.set").

    Parameters
    ----------
    in_dir : str or Path
        Root directory to begin searching from.

    folder_hints : list of str
        Ordered glob-style patterns. Each hint is applied one level at a time,
        progressively narrowing the search space.

    file_pattern : str
        Pattern used to match output files (e.g. "*.set", "*_eeg.set").

    Returns
    -------
    list of Path
        Sorted list of paths to files matching `file_pattern` inside the final
        matched folder layer.

    Notes
    -----
    • Uses `.rglob()` for recursive pattern matching.
    • Removes duplicates by storing files in a set.
    • Does not assume any particular directory standard (works for BIDS, EEG studies, etc.).
    """
    root = Path(in_dir)

    # ------------------------------------------------------------------
    # Start with the root as the only candidate
    # ------------------------------------------------------------------
    candidates: List[Path] = [root]

    # ------------------------------------------------------------------
    # Apply each folder hint in order
    # Each iteration reduces (filters) the candidate directory set
    # ------------------------------------------------------------------
    for hint in folder_hints:
        next_candidates: List[Path] = []

        for base_dir in candidates:
            # Search recursively under each candidate directory
            # for folders matching the hint (e.g., "sub-*", "ses-1", "eeg")
            for found_dir in base_dir.rglob(hint):
                if found_dir.is_dir():
                    next_candidates.append(found_dir)

        # Move to next stage of narrowing
        candidates = next_candidates

    # ------------------------------------------------------------------
    # Final stage: search only inside the last matched directories
    # ------------------------------------------------------------------
    file_results: set[Path] = set()

    for d in candidates:
        for f in d.rglob(file_pattern):  # recursive search for files
            file_results.add(f)          # ensures duplicates are removed

    # Return sorted list for reproducibility
    return sorted(file_results)


def _load_subject_mapping(
    metadata_path: str | Path,
    id_col: str = "UUID",
    label_col: str = "label",
) -> Optional[pd.DataFrame]:
    """
    Load a subject ID → label mapping from a user-supplied CSV or Excel file.

    This helper extracts only two columns from the metadata file:
        - `id_col`     → renamed to "subject_id"
        - `label_col`  → renamed to "label"

    The output is a small two-column DataFrame used for merging into the final
    EEG summary table.

    Parameters
    ----------
    metadata_path : str or Path
        Path to the metadata file (.csv, .xlsx, .xls).  
        If the file does not exist or cannot be opened, the function returns None.

    id_col : str
        Column name in the metadata file representing subject IDs.

    label_col : str
        Column name in the metadata file representing the label.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns:
            - "subject_id" : string
            - "label"  : string / int / bool (depends on user file)
        Returns None if:
            - File does not exist
            - File cannot be read
            - Required columns are missing

    Notes
    -----
    • This function performs almost no transformation so the user maintains full
      control over the shape and meaning of their metadata.
    • The calling function (scan_eeg_directory) handles the actual merge logic.
    """
    if metadata_path is None:
        # Caller enforces requirement; here we fail silently
        return None

    metadata_path = Path(metadata_path)

    # ------------------------------------------------------------------
    # Check file exists before attempting to read
    # ------------------------------------------------------------------
    if not metadata_path.exists():
        print(f"[warn] Metadata file '{metadata_path}' not found.")
        return None

    try:
        # ------------------------------------------------------------------
        # Read CSV or Excel depending on file extension
        # ------------------------------------------------------------------
        if metadata_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(metadata_path)
        else:
            df = pd.read_csv(metadata_path)

        # ------------------------------------------------------------------
        # Verify that both required columns exist
        # ------------------------------------------------------------------
        if {id_col, label_col}.issubset(df.columns):
            mapping = df[[id_col, label_col]].copy()
            mapping.rename(
                columns={id_col: "subject_id", label_col: "label"},
                inplace=True,
            )

            print(
                f"[info] Loaded mapping from '{metadata_path.name}' "
                f"({len(mapping)} rows, using columns '{id_col}' → 'subject_id', "
                f"'{label_col}' → 'label')."
            )
            return mapping

        # Missing columns → warn and return None
        missing = [c for c in (id_col, label_col) if c not in df.columns]
        print(f"[warn] '{metadata_path.name}' missing columns: {missing}")

    except Exception as e:
        print(f"[warn] Could not read metadata file '{metadata_path}': {e}")

    return None


def load_raw_eeg(p: str | Path, preload: bool = False) -> mne.io.BaseRaw:
    """
    Load an EEG file using the appropriate MNE-Python reader.

    This function performs lightweight validation of the file extension,
    selects the correct reader, and returns a Raw object without performing
    any additional processing.

    Parameters
    ----------
    p : str or Path
        Path to an EEG file. Supported formats include:
        .bdf, .edf, .cnt, .set, .fif, .mff, .egi, .raw, .gdf, .vhdr, .vmrk, .eeg, .cdt, .mat

    preload : bool
        Whether to preload the data into memory.
        Note: Some EEGLAB .set files *always* preload regardless of the value.

    Returns
    -------
    raw : mne.io.Raw
        The loaded MNE Raw object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    ValueError
        If the file extension is not known or unsupported.
    """
    p = Path(p).expanduser().resolve()

    # Ensure the file actually exists
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")

    ext = p.suffix.lower()

    # Mapping from file extension → MNE reader function
    loaders = {
        ".bdf": mne.io.read_raw_bdf,
        ".edf": mne.io.read_raw_edf,
        ".cnt": mne.io.read_raw_cnt,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
        ".mff": mne.io.read_raw_egi,
        ".egi": mne.io.read_raw_egi,
        ".raw": mne.io.read_raw_egi,
        ".gdf": mne.io.read_raw_gdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".vmrk": mne.io.read_raw_brainvision,
        ".eeg": mne.io.read_raw_brainvision,
        ".cdt": mne.io.read_raw_curry, 
        ".mat": mne.io.read_raw_fieldtrip,
    }

    # Check support
    if ext not in loaders:
        raise ValueError(f"Unsupported EEG extension: '{ext}'")

    # Select correct reader and load file
    if ext in ['.mat']:
        loader = loaders[ext]
        raw = loader(fname=p, info=None, data_name="ft" )
    else:
        loader = loaders[ext]
        raw = loader(p, preload=preload)

    return raw


def _rebuild_fieldtrip_raw_with_clean_info(
    raw: mne.io.BaseRaw,
    montage: mne.channels.DigMontage,
) -> mne.io.BaseRaw:
    """
    Rebuild a Raw object imported from a FieldTrip `.mat` file using a fresh
    MNE `Info`, then apply the requested montage.

    This helper is intended for Raw objects created via
    `mne.io.read_raw_fieldtrip(..., info=None)`. In that import path, MNE only
    extracts limited metadata, so channel type and sensor-location information
    may be incomplete or inconsistent. Rebuilding the Raw with a clean `Info`
    avoids montage-assignment failures caused by the imported metadata.

    The rebuilt object preserves:
    - channel names
    - sampling frequency
    - signal data
    - annotations, if present

    Channel typing rule
    -------------------
    - Channels whose names appear in `montage.ch_names` are assigned type
      `"eeg"`.
    - Channels not present in the montage are assigned type `"misc"`.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object loaded from a FieldTrip `.mat` file.

    montage : mne.channels.DigMontage
        Standard or custom montage to apply to the rebuilt Raw object.

    Returns
    -------
    mne.io.BaseRaw
        A new `RawArray` with fresh channel metadata and the montage applied.

    Notes
    -----
    This function is only needed for the FieldTrip import path. Other EEG
    readers usually provide more complete metadata and can typically use
    `raw.set_montage()` directly.
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    montage_chs = set(montage.ch_names)
    eeg_chs = [ch for ch in ch_names if ch in montage_chs]
    misc_chs = [ch for ch in ch_names if ch not in montage_chs]
    ch_types = ["eeg" if ch in montage_chs else "misc" for ch in ch_names]

    print(
        "[FieldTrip fix] Rebuilding Raw with clean Info before montage "
        f"({len(eeg_chs)} EEG channels, {len(misc_chs)} MISC channels)."
    )

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    new_raw = mne.io.RawArray(data, info)
    new_raw.set_montage(montage)

    if raw.annotations is not None and len(raw.annotations):
        new_raw.set_annotations(raw.annotations)

    print("[FieldTrip fix] Montage successfully applied after rebuild.")

    return new_raw


def load_abcct_mat_metadata(p: str | Path) -> Dict[str, Any]:
    """
    Load an ABC-CT-style .mat EEG file using h5py and extract lightweight metadata.

    This reader is specific to the ABC-CT format, which stores EEG data in HDF5
    with required datasets:
        - "EEG_Resting"     : 3-D array (segments × samples × channels)
        - "samplingRate"    : sampling frequency (scalar or 1×1 array)

    Parameters
    ----------
    p : str or Path
        Path to a .mat HDF5 file (ABC-CT EEG format).

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'keys'              : comma-separated string of HDF5 group keys
        - 'n_segments'        : number of EEG segments (int)
        - 'n_times'           : number of time samples per segment (int)
        - 'n_channels'        : number of channels (int)
        - 'sfreq_hz'          : sampling rate as float
        - 'seg_dur_second'    : duration (seconds) of a single segment
        - 'total_dur_second'  : total duration across all segments
        - 'error'             : None if OK, otherwise the error message (str)

    Notes
    -----
    • This function does *not* load EEG samples into memory.
    • It only inspects the HDF5 structure and extracts shape information.
    """
    p = Path(p)

    # Initialize output structure
    meta: Dict[str, Any] = {
        "keys": None,
        "n_segments": None,
        "n_times": None,
        "n_channels": None,
        "sfreq_hz": None,
        "seg_dur_second": None,
        "total_dur_second": None,
        "error": None,
    }

    try:
        # Open file in read mode
        with h5py.File(p, "r") as f:

            # Store top-level HDF5 keys
            keys = list(f.keys())
            meta["keys"] = ",".join(keys)

            # Verify required datasets exist
            if "EEG_Resting" not in f or "samplingRate" not in f:
                raise KeyError("Missing 'EEG_Resting' or 'samplingRate' dataset")

            # EEG data array: expected shape (segments × samples × channels)
            X = f["EEG_Resting"]
            if X.ndim != 3:
                raise ValueError(f"Unexpected EEG_Resting shape: {X.shape}")
            n_seg, n_times, n_ch = X.shape

            # Extract sampling rate (may be scalar or 1×1 array)
            sr = f["samplingRate"][()]
            sfreq = float(sr[0, 0]) if getattr(sr, "shape", None) == (1, 1) else float(sr)

            # Fill metadata outputs
            meta.update({
                "n_segments": n_seg,
                "n_times": n_times,
                "n_channels": n_ch,
                "sfreq_hz": sfreq,
                "seg_dur_second": n_times / sfreq,
                "total_dur_second": (n_times * n_seg) / sfreq,
            })

    except Exception as e:
        meta["error"] = str(e)

    return meta


def subject_id_from_borders(p: str | Path, borders: list[str] | None) -> str:
    """
    Extract a subject ID from a filename by cutting the *stem* at the position of
    the final “border” string in a user-provided sequence.

    This is a flexible, hint-based alternative to hardcoding `split("_")[0]`,
    and works well when datasets use different naming schemes.

    Behavior
    --------
    - Uses the filename *stem* (filename without extension).
    - If `borders` is None or empty: returns the full stem.
    - Otherwise, searches for each border in order, left-to-right.
      Each subsequent search begins *after* the previous match.
    - The subject ID returned is everything *before* the last matched border.
    - If any border is not found: falls back to returning the full stem.

    Examples
    --------
    stem = "NDARAH518DRB_20220915_r"
      borders=["_"]        -> "NDARAH518DRB"
      borders=["_","_"]    -> "NDARAH518DRB_20220915"

    stem = "sub-0046_task:rest}v1"
      borders=["_",":","}"] -> "sub-0046_task:rest"
    """
    # Get filename without extension (e.g., "NDARAH518DRB_20220915_r")
    stem = Path(p).stem

    # No borders/hints provided => treat whole stem as the subject ID
    if not borders:
        return stem

    idx = 0          # where the next search begins
    cut_pos = None   # position of the most recent matched border

    # Find each border in sequence, updating the search start each time
    for b in borders:
        pos = stem.find(b, idx)
        if pos == -1:
            # If any border is missing, do not partially cut—return full stem
            return stem

        cut_pos = pos
        idx = pos + len(b)  # continue searching after this border match

    # Cut right before the last border match
    return stem[:cut_pos] if cut_pos is not None else stem


def load_h5_keys(p: str | Path) -> dict:
    """
    Open an HDF5 file and return the top-level keys in a consistent format.

    Returns
    -------
    dict
        {
            "keys":      list[str],   # top-level keys as a list (canonical form)
            "keys_str":  str          # same keys joined by commas (handy for CSV/logging)
        }
    """
    p = Path(p)
    with h5py.File(p, "r") as f:
        keys = list(f.keys())

    return {
        "keys": keys,
    }

def explore_h5_key(p: str | Path, key: str) -> dict:
    """
    Explore a single top-level key in an HDF5 file.

    Returns lightweight info (no dataset values loaded):
    - If key is a group: list immediate children.
    - If key is a dataset: report shape and dtype.
    """
    p = Path(p)
    with h5py.File(p, "r") as f:
        if key not in f:
            return {"key": key, "error": "Key not found"}

        obj = f[key]

        if isinstance(obj, h5py.Group):
            return {
                "key": key,
                "type": "group",
                "children": list(obj.keys()),
            }

        if isinstance(obj, h5py.Dataset):
            return {
                "key": key,
                "type": "dataset",
                "shape": tuple(obj.shape),   # <- necessary cleanup
                "dtype": str(obj.dtype),
            }

        return {"key": key, "type": str(type(obj))}


def _unwrap_singletons(x: Any) -> Any:
    """
    Unwrap nested singleton containers.

    This is mainly for tiny HDF5 datasets that store scalars as 1x1 arrays,
    which often become nested Python lists after `.tolist()`.

    Examples
    --------
    [[1000.0]] -> 1000.0
    [1000.0]   -> 1000.0
    [[["a"]]]  -> "a"

    Notes
    -----
    - Only unwraps Python lists (not dicts, tuples, etc.).
    - Stops as soon as the value is not a single-item list.
    """
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


def read_h5_dataset_value(p: str | Path, key: str, max_elems: int = 16) -> Any | None:
    """
    Read a dataset value from an HDF5 file ONLY if it is small.

    This prevents accidentally loading large EEG arrays into memory.
    If the dataset is small enough, the value is read and then normalized
    via `_unwrap_singletons()` so scalar-like datasets do not come back as
    nested lists (e.g., [[1000.0]]).

    Parameters
    ----------
    p : str or Path
        Path to the HDF5 file.

    key : str
        Top-level dataset name (must exist in the file).

    max_elems : int
        Maximum number of elements allowed to read from the dataset.
        Datasets larger than this are skipped (returns None).

    Returns
    -------
    Any or None
        - Returns the dataset value if it is small enough.
        - Returns None if:
            - key does not exist
            - key is not a dataset
            - dataset has more than `max_elems` elements

    Examples
    --------
    - A sampling rate stored as (1,1) -> returns 1000.0 (not [[1000.0]])
    - A large EEG array -> returns None
    """
    p = Path(p)

    with h5py.File(p, "r") as f:
        # Key missing -> nothing to read
        if key not in f:
            return None

        ds = f[key]

        # Only datasets have values to read
        if not isinstance(ds, h5py.Dataset):
            return None

        # Too big -> skip (protects against reading EEG arrays)
        if ds.size > max_elems:
            return None

        # Safe to read (tiny dataset)
        val = ds[()]

        # Convert numpy scalars/arrays to plain Python objects/lists
        if hasattr(val, "tolist"):
            val = val.tolist()

        # Clean up scalar-like results
        return _unwrap_singletons(val)


def _normalize_subject_id(x: Any) -> str | None:
    """
    Normalize numeric-like subject IDs into a consistent string for merging.

    Goal
    ----
    Make IDs like the following equivalent:
      - "sub-0046" -> "46"
      - "0046"     -> "46"
      - 46         -> "46"

    Rules
    -----
    - Missing values -> None
    - Extract the first run of digits and convert to int (removes leading zeros).
    - If no digits are found, return the original string (trimmed).

    Notes
    -----
    This is intended for *numeric-like* subject IDs. If you have alphanumeric
    IDs where digits are not the identity (e.g., NDAR-style IDs), be careful:
    those would be reduced to their first digit run if you enable normalization.
    """
    # Handle missing values safely (pandas + numpy friendly)
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return None

    s = str(x).strip()
    if s == "":
        return None

    # Find the first digit run (works for sub-0046, 0046, 46, etc.)
    m = re.search(r"(\d+)", s)
    if not m:
        # No digits -> return the raw string (still useful for non-numeric IDs)
        return s

    # int(...) removes leading zeros, then cast back to string
    return str(int(m.group(1)))


def infer_eeg_dims_from_shape(
    shape: tuple[int, ...],
    *,
    max_channels: int = 600,
    max_segments: int = 10,
    max_times: int | None = None,
) -> dict[str, Any]:
    """
    Infer EEG dimension sizes from an array shape using a lightweight heuristic.

    This helper is meant for "shape-only" inference (e.g., from an HDF5 dataset)
    without loading EEG samples into memory.

    Heuristic
    ---------
    - 2D arrays:
        * Exactly one axis must be "channels-like" (<= max_channels).
        * The other axis is treated as "times".
    - 3D arrays:
        * Exactly one axis must be "segments-like" (<= max_segments).
        * Among the remaining two axes, exactly one must be "channels-like"
          (<= max_channels).
        * The last axis is treated as "times".
    - Any ambiguous case returns {}.

    Optional constraint
    -------------------
    max_times:
        If provided, the inferred time dimension must be <= max_times.
        This can help reject accidental matches when shapes are unusual.

    Parameters
    ----------
    shape : tuple[int, ...]
        Dataset shape (e.g., (n_segments, n_times, n_channels)).

    max_channels : int
        Upper bound for what counts as "channels-like".

    max_segments : int
        Upper bound for what counts as "segments-like" (for 3D arrays).

    max_times : int or None
        Optional upper bound for what counts as "times-like".

    Returns
    -------
    dict
        - If 2D and unambiguous:
            {"n_times": int, "n_channels": int}
        - If 3D and unambiguous:
            {"n_segments": int, "n_times": int, "n_channels": int}
        - Otherwise:
            {}
    """

    # Helper: enforce optional max_times constraint
    def times_ok(n: int) -> bool:
        return True if max_times is None else (n <= max_times)

    # ------------------------------------------------------------------
    # 2D case: (times, channels) or (channels, times)
    # ------------------------------------------------------------------
    if len(shape) == 2:
        a, b = shape

        # Identify which axis could plausibly be channels
        ch_axes = []
        if a <= max_channels:
            ch_axes.append(0)
        if b <= max_channels:
            ch_axes.append(1)

        # Must be uniquely identifiable
        if len(ch_axes) != 1:
            return {}

        ch_axis = ch_axes[0]
        time_axis = 1 - ch_axis

        n_channels = int(shape[ch_axis])
        n_times = int(shape[time_axis])

        # Optional sanity check on times
        if not times_ok(n_times):
            return {}

        return {"n_channels": n_channels, "n_times": n_times}

    # ------------------------------------------------------------------
    # 3D case: (segments, times, channels) in any axis order
    # ------------------------------------------------------------------
    if len(shape) == 3:
        dims = list(shape)

        # Find the segment axis (must be uniquely "small")
        seg_axes = [i for i, d in enumerate(dims) if d <= max_segments]
        if len(seg_axes) != 1:
            return {}

        seg_axis = seg_axes[0]
        remaining = [i for i in range(3) if i != seg_axis]

        # Among remaining axes, find channels axis (must be uniquely "small")
        ch_axes = [i for i in remaining if dims[i] <= max_channels]
        if len(ch_axes) != 1:
            return {}

        ch_axis = ch_axes[0]
        time_axis = [i for i in remaining if i != ch_axis][0]

        n_segments = int(dims[seg_axis])
        n_channels = int(dims[ch_axis])
        n_times = int(dims[time_axis])

        # Optional sanity check on times
        if not times_ok(n_times):
            return {}

        return {
            "n_segments": n_segments,
            "n_channels": n_channels,
            "n_times": n_times,
        }

    # ------------------------------------------------------------------
    # Any other dimensionality -> not handled by this heuristic
    # ------------------------------------------------------------------
    return {}


def compute_durations(meta: dict) -> dict:
    """
    Compute seg_dur_second and total_dur_second if possible.

    Requires:
      - n_times
      - samplingRate_value (scalar)
      - optional n_segments (defaults to 1)
    """
    if "n_times" not in meta or "samplingRate_value" not in meta:
        return {}

    sfreq = meta["samplingRate_value"]
    if not isinstance(sfreq, (int, float)) or sfreq <= 0:
        return {}

    n_times = meta["n_times"]
    if not isinstance(n_times, int) or n_times <= 0:
        return {}

    n_segments = meta.get("n_segments", 1)
    if not isinstance(n_segments, int) or n_segments <= 0:
        n_segments = 1

    seg_dur = n_times / float(sfreq)
    total_dur = (n_times * n_segments) / float(sfreq)

    return {
        "seg_dur_second": seg_dur,
        "total_dur_second": total_dur,
    }


def _boundary_contains(needle: str, haystack: str) -> bool:
    # needle appears as a whole token-ish unit: boundaries are non-alnum or string edges
    # Works for underscores, hyphens, dots, slashes, spaces, etc.
    pat = re.compile(rf"(^|[^A-Za-z0-9]){re.escape(needle)}([^A-Za-z0-9]|$)")
    return pat.search(haystack) is not None

def resolve_to_metadata_id(file_id: str, meta_ids: list[str]) -> str | None:
    if file_id in meta_ids:
        return file_id

    candidates = [mid for mid in meta_ids if _boundary_contains(mid, file_id)]
    if not candidates:
        # optional: also allow reverse containment (rare but sometimes metadata is decorated)
        candidates = [mid for mid in meta_ids if _boundary_contains(file_id, mid)]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Prefer longest (most specific) only if uniquely longest
        candidates.sort(key=len, reverse=True)
        if len(candidates) >= 2 and len(candidates[0]) == len(candidates[1]):
            return None  # ambiguous
        return candidates[0]

    return None



def scan_eeg_directory(
    in_dir: str | Path = "abcct_data_raw",
    pattern: str = "*_r.mat",
    metadata_path: str | Path | None = None,
    id_col: str | None = "UUID",
    label_col: str | None = "label",
    folder_hints: Optional[list[str]] = None,
    backend: Literal["h5", "mne"] = "h5",
    subject_id_borders: list[str] | None = ["_"],
    normalize_numeric_subject_ids: bool = False,
    # ---- EEG dimension inference knobs (used for H5 inference) ----
    max_channels: int = 600,
    max_segments: int = 10,
    max_times: int | None = None,
    # ---- Small H5 value reading knob ----
    max_value_elems: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Scan EEG files inside a directory (optionally with folder hints) and return
    a summary table containing lightweight metadata for each file, merged with
    label metadata from a CSV/XLSX mapping file.

    This function is designed to be flexible across datasets where:
      - folder structures differ (use folder_hints)
      - filenames differ (use subject_id_borders)
      - HDF5 top-level keys differ (loop keys dynamically; no key name hardcoding)

    Many datasets have non-informative filenames like "EEG1.bdf". In those cases,
    we cannot derive the Subject_ID from the filename. We now add a robust rule:

      If any Subject_ID from the metadata file appears in the *full file path*
      (boundary-aware token match), we override the file-derived subject_id with
      that metadata Subject_ID.

    This keeps the old behavior working (when filenames contain IDs like CU0009),
    while also supporting datasets where the ID is only present in folder names.

    Parameters
    ----------
    in_dir : str or Path
        Root directory to start searching from.

    pattern : str
        File pattern to match, e.g. "*_r.mat", "*_eeg.set".

        
    metadata_path : str, Path, or None
        Optional path to a CSV/XLSX metadata file containing subject IDs
        and labels.

        If provided, the metadata file is used to match subject IDs and
        attach label information.

        If None, EEG files are still scanned normally and subject IDs are
        derived from the filename/path logic. No metadata merge or label
        assignment is performed.

    id_col : str or None
        Column name in the metadata file containing subject IDs.
        Required only when metadata_path is provided.

    label_col : str or None
        Column name in the metadata file containing the label / phenotype.
        Required only when metadata_path is provided.

    folder_hints : list of str, optional
        Successive folder filters (e.g., ["sub-*","ses-1","eeg"]) that narrow where
        to look before matching files by pattern.

    backend : {'h5', 'mne'}
        Determines how to load/parse EEG files.
        - "h5": HDF5-based files (e.g., ABC-CT .mat stored as HDF5). We dynamically
                loop top-level keys to infer EEG dims and read small scalar values.
        - "mne": EEG files supported by MNE (e.g. .set, .edf, .bdf).

    subject_id_borders : list of str, optional
        Ordered “border” strings used to extract `subject_id` from each EEG filename
        stem (filename without extension) via `subject_id_from_borders()`.

        Rules:
        - If None or empty: the full stem is used as the subject_id.
        - Otherwise: each border is searched in sequence; the subject_id becomes the
          portion of the stem before the *final* matched border.
        - If any border is not found: falls back to using the full stem.

        Examples:
        - "NDARAH518DRB_20220915_r.mat" with ["_"]       → "NDARAH518DRB"
        - "NDARAH518DRB_20220915_r.mat" with ["_","_"]   → "NDARAH518DRB_20220915"
        - "sub-0046_task:rest}v1.set" with ["_",":","}"] → "sub-0046_task:rest"

    normalize_numeric_subject_ids : bool, optional
        If True, normalize numeric-like subject IDs for merging using
        `_normalize_subject_id()` on BOTH:
          - metadata subject IDs (from `id_col`)
          - file-derived `subject_id` (from filenames)

        This is intended for IDs like: 'sub-0046', '0046', 46  -> '46'
        If False, subject IDs are used as-is (after string casting).

    max_channels : int
        Heuristic threshold for identifying the channels dimension when inferring
        EEG dimensions from an HDF5 dataset shape.

    max_segments : int
        Heuristic threshold for identifying the segments dimension when inferring
        EEG dimensions from an HDF5 dataset shape.

    max_times : int or None
        Optional threshold for identifying the times dimension when inferring EEG
        dimensions from an HDF5 dataset shape. If None, no upper bound is applied.

    max_value_elems : int
        Maximum number of elements allowed when reading a dataset value from HDF5.
        This prevents accidentally loading large arrays; only small datasets
        (e.g., scalar sampling rate stored as 1x1) are read.

    Returns
    -------
    df : pd.DataFrame
        One row per EEG file (matched + unmatched). Columns are built dynamically:
        - Always: filepath, filename, subject_id_raw, subject_id, subject_id_merge, error
        - H5: keys (list), inferred dims (n_times/n_channels[/n_segments]) when found,
            plus any small dataset values as "<key>_value", and durations when possible
        - MNE: fields extracted from the Raw object (as defined in your loader)
        - Metadata: merged label column ("label"), plus match_status (human-readable)

    uunmatched_df : pd.DataFrame
        Subset of df containing only files that did not match metadata
        (match_status == "unmatched_file_id"). Includes key ID/debug columns.

    """

    """
    Scan EEG files inside a directory (optionally with folder hints) and return
    a summary table containing lightweight metadata for each file, merged with
    label metadata from a CSV/XLSX mapping file.

    UPDATE (Feb 2026):
    ------------------
    Many datasets have non-informative filenames like "EEG1.bdf". In those cases,
    we cannot derive the Subject_ID from the filename. We now add a robust rule:

      If any Subject_ID from the metadata file appears in the *full file path*
      (boundary-aware token match), we override the file-derived subject_id with
      that metadata Subject_ID.

    This keeps the old behavior working (when filenames contain IDs like CU0009),
    while also supporting datasets where the ID is only present in folder names.

    Returns
    -------
    (df, uunmatched_df)
      - df: one row per EEG file
      - uunmatched_df: subset of df with match_status == "unmatched_file_id"
    """


    # ----------------------------------------------------------------------
    # 0) Validate optional metadata configuration
    # ----------------------------------------------------------------------
    metadata_provided = metadata_path is not None

    if metadata_provided:
        if id_col is None:
            raise ValueError(
                "id_col must be provided when metadata_path is supplied."
            )

        if label_col is None:
            raise ValueError(
                "label_col must be provided when metadata_path is supplied."
            )

    # Ensure Path object for filesystem operations
    in_dir = Path(in_dir)

    # We'll collect one dict per file and build the DataFrame at the end
    rows: list[dict] = []

    # ----------------------------------------------------------------------
    # 1) Locate EEG files: use folder hints if provided, else simple recursive
    # ----------------------------------------------------------------------
    if folder_hints:
        # Hint-based search (great for BIDS-like structures / irregular trees)
        files = find_files_with_hints(
            in_dir=in_dir,
            folder_hints=folder_hints,
            file_pattern=pattern,
        )
    else:
        # Plain recursive glob
        files = sorted(in_dir.rglob(pattern))

    # Nothing found -> return empty DataFrames early (must return 2 items)
    if not files:
        print(f"No files found in {in_dir} with pattern='{pattern}' and backend='{backend}'.")
        return pd.DataFrame(), pd.DataFrame()

    # ----------------------------------------------------------------------
    # 2) Load metadata mapping (subject_id + label)
    # ----------------------------------------------------------------------
    # subj_map = _load_subject_mapping(metadata_path, id_col=id_col, label_col=label_col)
    # if subj_map is None:
    #     raise ValueError(f"Could not load metadata file: {metadata_path}")

    # # Convert to pandas "string" dtype (keeps missing values as <NA>, not "nan")
    # subj_map["subject_id"] = subj_map["subject_id"].astype("string")

    # # Optional: normalize numeric-like IDs (only if you turn it on)
    # if normalize_numeric_subject_ids:
    #     subj_map["subject_id"] = subj_map["subject_id"].map(_normalize_subject_id).astype("string")

    # # Precompute metadata IDs as plain strings for path-matching override
    # # (this is the "ground truth" list of valid Subject_ID tokens)
    # meta_ids: list[str] = subj_map["subject_id"].dropna().astype(str).unique().tolist()



    # ----------------------------------------------------------------------
    # 2) Optionally load metadata mapping (subject_id + label)
    # ----------------------------------------------------------------------
    subj_map = None
    meta_ids: list[str] = []

    if metadata_provided:
        subj_map = _load_subject_mapping(  metadata_path,  id_col=id_col,label_col=label_col, )

        if subj_map is None:
            raise ValueError( f"Could not load metadata file: {metadata_path}" )

        # Convert to pandas string dtype
        subj_map["subject_id"] = ( subj_map["subject_id"].astype("string"))

        # Optional normalization of numeric-like IDs
        if normalize_numeric_subject_ids:
            subj_map["subject_id"] = (subj_map["subject_id"].map(_normalize_subject_id).astype("string"))

        # Ground-truth metadata IDs used for path-based matching
        meta_ids = subj_map["subject_id"].dropna().astype(str).unique().tolist()

    else:
        print(
            "[info] No metadata file provided. "
            "Scanning EEG files without subject-label matching."
        )


    # ----------------------------------------------------------------------
    # 3) Process each EEG file
    # ----------------------------------------------------------------------
    for p in tqdm(files, desc="Scanning EEG files", unit="file"):

        # --------------------------
        # A) Default: derive ID from filename stem using borders (old behavior)
        # --------------------------
        file_id_raw = subject_id_from_borders(p, subject_id_borders)

        info = {
            "filepath": str(p),
            "filename": p.name,
            "subject_id_raw": file_id_raw,     # raw derived id (debug)
            "subject_id": file_id_raw,         # working id for merge
            "subject_id_source": "filename",   # NEW: where subject_id came from
            "error": None,
        }

        # Optional: normalize file-derived IDs too (must match metadata side)
        if normalize_numeric_subject_ids:
            info["subject_id"] = _normalize_subject_id(info["subject_id"])

        # --------------------------
        # B) NEW: Path-based override using metadata Subject_ID list
        # --------------------------
        # If a metadata Subject_ID appears in the full file path, prefer it.
        # This fixes cases like ".../CU_CU0009_1R1_eeg/EEG1.bdf".
        #
        # We only do this override when NOT using numeric normalization mode,
        # because normalization mode expects separate logic for numeric-like IDs.
        # if not normalize_numeric_subject_ids:
            # Resolve a unique metadata id from the full path (boundary-aware).
            # Uses your existing helper logic style (safe vs false positives).
            # path_match = resolve_to_metadata_id(str(p), meta_ids)

        if metadata_provided and not normalize_numeric_subject_ids:
            path_match = resolve_to_metadata_id(str(p),meta_ids)

            if path_match is not None:
                info["subject_id_raw"] = path_match
                info["subject_id"] = path_match
                info["subject_id_source"] = "path_match"

        # C) Refined: ONLY prepend subject_id if it's not already in the original filename
        #    - case-insensitive
        #    - boundary-aware via existing _boundary_contains helper
        subj = str(info["subject_id"]) if info["subject_id"] is not None else ""
        orig_name = p.name

        # Case-insensitive token match: compare using lowercased strings
        has_id_in_name = bool(subj) and _boundary_contains(subj.lower(), orig_name.lower())

        if not has_id_in_name:
            info["filename"] = f"{subj}_{orig_name}" if subj else orig_name
        else:
            info["filename"] = orig_name  # keep as-is

        try:

            # --------------------------
            # H5 backend: dynamic key loop
            # --------------------------
            if backend == "h5":
                # Always safe/lightweight: list top-level keys
                info.update(load_h5_keys(p))  # {"keys": [...]}

                # Loop through EVERY key and:
                #  1) infer EEG dims from dataset shapes (2D or 3D)
                #  2) read small dataset values into "<key>_value"
                for k in info.get("keys", []):
                    d = explore_h5_key(p, k)

                    # We only care about datasets here (groups have no shapes/values)
                    if d.get("type") != "dataset":
                        continue

                    # ---- 1) Infer EEG dims from shape (no key-name hardcoding) ----
                    shape = d.get("shape")
                    if shape:
                        dims = infer_eeg_dims_from_shape(
                            shape,
                            max_channels=max_channels,
                            max_segments=max_segments,
                            max_times=max_times,
                        )

                        # Store dims ONCE (first match wins)
                        if dims and not any(key in info for key in ("n_times", "n_channels", "n_segments")):
                            info.update(dims)

                    # ---- 2) Read small dataset values (scalar-ish) ----
                    val = read_h5_dataset_value(p, k, max_elems=max_value_elems)
                    if val is not None:
                        info[f"{k}_value"] = val

                # Compute duration after we've collected dims + samplingRate_value
                info.update(compute_durations(info))

            # --------------------------
            # MNE backend
            # --------------------------
            elif backend == "mne":
                meta = load_raw_eeg(p, preload=False)

                # If your load_raw_eeg already returns a dict, this is fine:
                if isinstance(meta, dict):
                    info.update(meta)
                else:
                    # Minimal inline extraction (only if meta is an MNE Raw-like object)
                    raw = meta
                    n_times = int(raw.n_times)
                    sfreq = float(raw.info["sfreq"])
                    n_ch = int(raw.info["nchan"])
                    duration = float(n_times / sfreq)

                    info.update({
                        "n_times": n_times,
                        "n_channels": n_ch,
                        "sfreq_hz": sfreq,
                        "seg_dur_second": duration,
                        "total_dur_second": duration,
                    })

            else:
                # Unknown backend -> record error on that row
                info["error"] = f"Unknown backend: {backend}"

        except Exception as e:
            # Catch errors per-file so one bad file doesn't kill the whole scan
            info["error"] = str(e)

        rows.append(info)

    # Convert to DataFrame (pandas will handle missing keys as NaN)
    df = pd.DataFrame(rows)


    # ----------------------------------------------------------------------
    # 4) Merge label metadata
    # ----------------------------------------------------------------------
    uunmatched_df = pd.DataFrame()  # ensure defined even if df empty

    if not df.empty:
        # Ensure consistent dtype
        df["subject_id"] = df["subject_id"].astype("string")

        # ------------------------------------------------------------------
        # A) Metadata was provided
        # ------------------------------------------------------------------
        if subj_map is not None:
            subj_map["subject_id"] = subj_map["subject_id"].astype("string")

            # Build a canonical merge key
            if normalize_numeric_subject_ids:
                df["subject_id_merge"] = df["subject_id"].map(_normalize_subject_id).astype("string")
                subj_map["subject_id_merge"] = subj_map["subject_id"].map(_normalize_subject_id).astype("string")
            else:
                meta_ids = subj_map["subject_id"].dropna().astype(str).unique().tolist()
                df["subject_id_merge"] = (
                    df["subject_id"].astype(str)
                    .map(lambda s: resolve_to_metadata_id(s, meta_ids))
                    .astype("string")
                )
                subj_map["subject_id_merge"] = subj_map["subject_id"].astype("string")

            # Safety: one row per subject in metadata
            subj_map = subj_map.drop_duplicates(subset=["subject_id_merge"], keep="first")

            # Exact merge
            df = df.merge(
                subj_map.drop(columns=["subject_id"]),
                on="subject_id_merge",
                how="left",
                validate="many_to_one",
                indicator=True,
            )

            # Convert missing merge values from <NA> to np.nan
            df["subject_id_merge"] = (
                df["subject_id_merge"]
                .astype(object)
                .where(df["subject_id_merge"].notna(), np.nan)
            )

            # Human-friendly match status
            df["match_status"] = df["_merge"].map({
                "both": "matched",
                "left_only": "unmatched_file_id",
                "right_only": "metadata_only",
            }).astype("string")

            # Summary
            total = len(df)
            matched = int((df["_merge"] == "both").sum())
            unmatched = int((df["_merge"] == "left_only").sum())
            meta_only = int((df["_merge"] == "right_only").sum())

            print(
                f"[info] Label merge summary: {matched}/{total} files matched "
                f"({matched/total:.1%}). Unmatched file IDs: {unmatched}. "
                f"Metadata-only IDs: {meta_only}."
            )
            print("[info] match_status counts:")
            print(df["match_status"].value_counts(dropna=False))

            uunmatched_df = df.loc[
                df["match_status"] == "unmatched_file_id",
                ["subject_id_raw", "subject_id", "subject_id_merge", "filename", "filepath", "subject_id_source"],
            ].copy()

        # ------------------------------------------------------------------
        # B) No metadata was provided
        # ------------------------------------------------------------------
        else:
            df["subject_id_merge"] = df["subject_id"].astype("string")
            df["label"] = np.nan
            df["match_status"] = "metadata_not_provided"

            print(
                f"[info] Scanned {len(df)} EEG file(s) "
                "without external metadata."
            )

    return df, uunmatched_df

    # # ----------------------------------------------------------------------
    # # 4) Merge label metadata
    # # ----------------------------------------------------------------------
    # uunmatched_df = pd.DataFrame()  # ensure defined even if df empty

    # if not df.empty:

    #     # Ensure consistent dtype for safe merging (<NA> preserved)
    #     df["subject_id"] = df["subject_id"].astype("string")
    #     subj_map["subject_id"] = subj_map["subject_id"].astype("string")

    #     # Build a canonical merge key
    #     if normalize_numeric_subject_ids:
    #         df["subject_id_merge"] = df["subject_id"].map(_normalize_subject_id).astype("string")
    #         subj_map["subject_id_merge"] = subj_map["subject_id"].map(_normalize_subject_id).astype("string")
    #     else:
    #         # NOTE:
    #         # We keep this logic, but now df["subject_id"] is often already the
    #         # exact metadata Subject_ID thanks to the path-based override above.
    #         meta_ids = subj_map["subject_id"].dropna().astype(str).unique().tolist()
    #         df["subject_id_merge"] = (
    #             df["subject_id"].astype(str)
    #             .map(lambda s: resolve_to_metadata_id(s, meta_ids))
    #             .astype("string")
    #         )
    #         subj_map["subject_id_merge"] = subj_map["subject_id"].astype("string")

    #     # Safety: one row per subject in metadata
    #     subj_map = subj_map.drop_duplicates(subset=["subject_id_merge"], keep="first")

    #     # Exact merge (validated)
    #     df = df.merge(
    #         subj_map.drop(columns=["subject_id"]),  # prevent duplicate cols
    #         on="subject_id_merge",
    #         how="left",
    #         validate="many_to_one",
    #         indicator=True,
    #     )

    #     # Convert subject_id_merge missing values to np.nan (instead of <NA>)
    #     df["subject_id_merge"] = (
    #         df["subject_id_merge"]
    #         .astype(object)
    #         .where(df["subject_id_merge"].notna(), np.nan)
    #     )

    #     # Human-friendly match status
    #     df["match_status"] = df["_merge"].map({
    #         "both": "matched",
    #         "left_only": "unmatched_file_id",
    #         "right_only": "metadata_only",
    #     }).astype("string")

    #     # Clearer summary printout
    #     total = len(df)
    #     matched = int((df["_merge"] == "both").sum())
    #     unmatched = int((df["_merge"] == "left_only").sum())
    #     meta_only = int((df["_merge"] == "right_only").sum())

    #     print(
    #         f"[info] Label merge summary: {matched}/{total} files matched "
    #         f"({matched/total:.1%}). Unmatched file IDs: {unmatched}. "
    #         f"Metadata-only IDs: {meta_only}."
    #     )

    #     print("[info] match_status counts:")
    #     print(df["match_status"].value_counts(dropna=False))

    #     uunmatched_df = df.loc[
    #         df["match_status"] == "unmatched_file_id",
    #         ["subject_id_raw", "subject_id", "subject_id_merge", "filename", "filepath", "subject_id_source"],
    #     ].copy()

    # return df, uunmatched_df


def convert_scan_to_mne_fif(
    df: pd.DataFrame,
    out_dir: str | Path = "sample_butler_prepared",
    montage_name: str = "biosemi64",
    overwrite: bool = False,
    preload: bool = True,
    *,
    metadata_path: str | Path | None = None,
    id_col: str | None = "id",
    label_col: str | None = "label",
    strict: bool = True,
    normalize_numeric_subject_ids: bool = False,
    channel_types: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert MNE-readable EEG files listed in `df` into standardized MNE FIF files.

    If metadata_path is provided, metadata is used as the source of truth for
    subject inclusion and labels. If metadata_path is None, all scanned files
    are retained and assigned to "UNLABELED" when no label is available.

    Optional channel_types can explicitly correct channel classifications before
    saving, for example {"HEOG": "eog", "VEOG": "eog", "Trigger": "misc"}.

    Parameters
    ----------
    df : pd.DataFrame
        Scan table containing at least filepath, filename, and subject_id.

    out_dir : str | Path
        Root output directory. FIF files are written under:
            out_dir/<label>/<filename_stem>_eeg.fif

    montage_name : str
        Name of the MNE standard montage to apply.

    overwrite : bool
        Whether to overwrite existing FIF files.

    preload : bool
        Passed to the EEG loader.

    metadata_path : str, Path, or None
        Optional CSV/XLSX metadata file containing subject IDs and labels.

    id_col : str or None
        Subject-ID column in metadata. Required only if metadata_path is supplied.

    label_col : str or None
        Label column in metadata. Required only if metadata_path is supplied.

    strict : bool
        If True, raise an error when metadata filtering leaves no files.

    normalize_numeric_subject_ids : bool
        If True, normalize numeric-like subject IDs before matching.

    channel_types : Mapping[str, str] or None
        Optional channel-name to MNE channel-type mapping.

        Example:
            {
                "HEOG": "eog",
                "VEOG": "eog",
                "Trigger": "misc",
            }

    Returns
    -------
    converted_df : pd.DataFrame
        Rows selected for conversion.

    uunmatched_df : pd.DataFrame
        Rows that could not be matched to supplied metadata. Empty when
        metadata_path is None.
    """
    # ------------------------------------------------------------------
    # 0) Validate input
    # ------------------------------------------------------------------
    required_cols = {"filepath", "subject_id", "filename"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[convert_scan_to_mne_fif] df is missing required columns: {sorted(missing_cols)}"
        )
    if channel_types is not None and not isinstance(channel_types, Mapping):
        raise TypeError("channel_types must be a mapping or None.")

    work = df.copy()
    uunmatched_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # 1) Prepare rows for conversion
    # ------------------------------------------------------------------
    if metadata_path is None:
        # No external metadata: retain all scanned EEG files
        converted_df = work.copy()
        converted_df["subject_id_key"] = converted_df["subject_id"].astype("string")

        # Preserve existing labels when available; otherwise use UNLABELED
        if "label" not in converted_df.columns:
            converted_df["label"] = "UNLABELED"
        else:
            converted_df["label"] = converted_df["label"].astype("string").fillna("").str.strip()
            converted_df.loc[converted_df["label"] == "", "label"] = "UNLABELED"

        print(
            f"[convert_scan_to_mne_fif] No metadata provided. "
            f"Keeping all {len(converted_df)} file(s)."
        )

    else:
        # Metadata supplied: use it as the source of truth
        if id_col is None:
            raise ValueError("id_col must be provided when metadata_path is supplied.")
        if label_col is None:
            raise ValueError("label_col must be provided when metadata_path is supplied.")

        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"[convert_scan_to_mne_fif] metadata_path not found: {metadata_path}"
            )

        # Load CSV or Excel metadata
        if metadata_path.suffix.lower() in {".xlsx", ".xls"}:
            meta = pd.read_excel(metadata_path)
        else:
            meta = pd.read_csv(metadata_path)

        if id_col not in meta.columns or label_col not in meta.columns:
            raise ValueError(
                f"[convert_scan_to_mne_fif] metadata file must contain columns "
                f"'{id_col}' and '{label_col}'. Found: {list(meta.columns)}"
            )

        meta = meta[[id_col, label_col]].copy()

        # Build canonical metadata subject key
        if normalize_numeric_subject_ids:
            meta["subject_id_key"] = meta[id_col].map(_normalize_subject_id)
        else:
            meta["subject_id_key"] = meta[id_col].astype(str).str.strip()

        meta["label_meta"] = meta[label_col]

        # Remove unusable metadata rows
        meta = meta.dropna(subset=["subject_id_key"])
        meta["label_meta"] = meta["label_meta"].apply(
            lambda x: np.nan if pd.isna(x) else x
        )
        meta = meta.dropna(subset=["label_meta"])
        meta = meta[meta["label_meta"].astype(str).str.strip() != ""]

        # Warn if numeric normalization collapses multiple raw IDs
        if normalize_numeric_subject_ids:
            n_raw_per_key = meta.groupby("subject_id_key")[id_col].nunique()
            collisions = n_raw_per_key[n_raw_per_key > 1]
            if not collisions.empty:
                print(
                    "[convert_scan_to_mne_fif] Warning: numeric normalization "
                    "caused subject-ID collisions."
                )
                print(collisions.head(10))

        # One metadata row per canonical subject ID
        meta = meta.drop_duplicates(subset=["subject_id_key"], keep="first")

        # Build scan-table subject key
        if normalize_numeric_subject_ids:
            work["subject_id_key"] = work["subject_id"].map(_normalize_subject_id)
        else:
            meta_ids = meta["subject_id_key"].dropna().astype(str).unique().tolist()
            work["subject_id_key"] = (
                work["subject_id"]
                .astype(str)
                .str.strip()
                .map(lambda s: resolve_to_metadata_id(s, meta_ids))
            )

        # Store files that cannot be matched to metadata
        uunmatched_df = work.loc[
            work["subject_id_key"].isna()
            | ~work["subject_id_key"].isin(meta["subject_id_key"]),
            ["subject_id", "subject_id_key", "filename", "filepath"],
        ].copy()

        # Keep only files represented in metadata
        converted_df = work.merge(
            meta[["subject_id_key", "label_meta"]],
            on="subject_id_key",
            how="inner",
            validate="many_to_one",
        )

        # Metadata label is the source of truth
        converted_df["label"] = converted_df["label_meta"]
        converted_df = converted_df.drop(columns=["label_meta"])

        n_in = len(df)
        n_keep = len(converted_df)
        n_drop = n_in - n_keep
        print(
            f"[convert_scan_to_mne_fif] Keeping {n_keep}/{n_in} files after "
            f"metadata filter. Dropped {n_drop}."
        )

        if strict and n_keep == 0:
            raise ValueError(
                "[convert_scan_to_mne_fif] After metadata filtering, "
                "no rows remain to convert."
            )

    # ------------------------------------------------------------------
    # 2) Prepare output directory and montage
    # ------------------------------------------------------------------
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        montage = mne.channels.make_standard_montage(montage_name)
    except Exception as exc:
        raise ValueError(
            f"[convert_scan_to_mne_fif] Invalid montage '{montage_name}'."
        ) from exc

    # ------------------------------------------------------------------
    # 3) Convert EEG files to FIF
    # ------------------------------------------------------------------
    failed: list[tuple[str, str]] = []

    for row in tqdm(
        converted_df.itertuples(index=False),
        total=len(converted_df),
        desc="Converting EEG → FIF (by label)",
        unit="file",
    ):
        file_path = Path(row.filepath)
        filename = str(row.filename)
        label = str(row.label).strip()

        # Create output folder
        label_dir = out_dir_path / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Use MNE-compatible EEG filename convention
        stem = Path(filename).stem
        out_path = label_dir / f"{stem}_eeg.fif"

        if out_path.exists() and not overwrite:
            continue

        try:
            # Load using the general MNE-compatible EEG loader
            raw = load_raw_eeg(file_path, preload=preload)

            # FieldTrip MAT requires rebuilding clean MNE channel information
            if file_path.suffix.lower() == ".mat":
                raw = _rebuild_fieldtrip_raw_with_clean_info(raw, montage)

            # Apply explicitly requested channel types
            if channel_types:
                present_types = {
                    ch: ch_type
                    for ch, ch_type in channel_types.items()
                    if ch in raw.ch_names
                }
                missing_types = [
                    ch for ch in channel_types
                    if ch not in raw.ch_names
                ]

                if present_types:
                    raw.set_channel_types(present_types)

                if missing_types:
                    print(
                        "[convert_scan_to_mne_fif] "
                        f"Channel-type names not found, skipping: {missing_types}"
                    )

            # Non-MAT formats: standardize EEG channel locations
            if file_path.suffix.lower() != ".mat":
                montage_chs = set(montage.ch_names)
                current_types = dict(
                    zip(raw.ch_names, raw.get_channel_types())
                )

                # Only demote channels that are still typed as EEG but are not
                # present in the requested montage. EOG/misc/stim types remain intact.
                unknown_eeg = [
                    ch for ch in raw.ch_names
                    if ch not in montage_chs and current_types[ch] == "eeg"
                ]

                if unknown_eeg:
                    raw.set_channel_types({
                        ch: "misc"
                        for ch in unknown_eeg
                    })

                # Replace source coordinates with the requested standard montage
                raw.set_montage(montage)

            # Save standardized FIF while preserving annotations
            raw.save(
                out_path.as_posix(),
                overwrite=overwrite,
                verbose=False,
            )

        except Exception as e:
            failed.append((file_path.name, str(e)))

    # ------------------------------------------------------------------
    # 4) Summary
    # ------------------------------------------------------------------
    if failed:
        print("\n[convert_scan_to_mne_fif] Failed to process the following files:")
        for fname, err in failed:
            print(f" - {fname}: {err}")

    return converted_df, uunmatched_df



# def convert_scan_to_mne_fif(
#     df: pd.DataFrame,
#     out_dir: str | Path = "sample_butler_prepared",
#     montage_name: str = "biosemi64",
#     overwrite: bool = False,
#     preload: bool = True,
#     *,
#     metadata_path: str | Path | None = None,
#     id_col: str | None = "id",
#     label_col: str | None = "label",
#     strict: bool = True,
#     normalize_numeric_subject_ids: bool = False,  
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Convert MNE-readable EEG files listed in a scan DataFrame into MNE FIF files.

#     Supported source formats are determined by `load_raw_eeg()` and may include
#     BDF, EDF, CNT, EEGLAB SET, BrainVision, Curry CDT, FIF, GDF, EGI, and
#     FieldTrip MAT files.

#     The converted FIF files are organized into output subfolders by metadata label.

#     This function uses the metadata file as the source of truth for:
#       1) which subjects are eligible for conversion, and
#       2) what label each subject should receive.

#     ID matching modes
#     -----------------
#     normalize_numeric_subject_ids = False (default)
#         Treat metadata IDs as canonical string tokens. File-derived IDs may be
#         "decorated" (prefix/suffix/date/run markers). We resolve each file ID to
#         a metadata ID using `resolve_to_metadata_id()` (boundary-aware containment),
#         then do an exact join.

#     normalize_numeric_subject_ids = True
#         Intended for numeric-like IDs (e.g., 46, "0046", "sub-0046"). We normalize
#         BOTH metadata and file IDs via `_normalize_subject_id()` and join exactly.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Scan table describing input EEG files. Must include at least:
#         - 'filepath'   : path to the source `.set` file
#         - 'filename'   : basename (used for output naming)
#         - 'subject_id' : ID extracted during the scan step

#     out_dir : str | Path
#         Root output directory. Output FIF files are written under:
#             out_dir/<label>/<filename_stem>.fif

#     montage_name : str
#         Name of an MNE standard montage to apply (default "biosemi64"). Channels
#         not present in the montage are re-typed as 'misc' so they are retained.

#     overwrite : bool
#         If True, overwrite existing `.fif` files. If False, existing outputs
#         are skipped.

#     preload : bool
#         Passed to the EEG loader when reading `.set` files.

#     metadata_path : str | Path
#         Path to a CSV/XLSX metadata file containing canonical subject IDs and labels.

#     id_col : str
#         Column name in the metadata file containing subject IDs.

#     label_col : str
#         Column name in the metadata file containing labels.

#     strict : bool
#         If True, raise an error if filtering by metadata results in zero rows to
#         convert. If False, the function returns empty outputs after printing a summary.

#     normalize_numeric_subject_ids : bool
#         Controls ID matching mode (see "ID matching modes" above).

#     Returns
#     -------
#     converted_df : pd.DataFrame
#         Subset of input `df` that matched metadata, with columns:
#         - subject_id_key : canonical key used for metadata join
#         - label          : label from metadata (source of truth)

#     uunmatched_df : pd.DataFrame
#         Rows from `df` that could not be matched to metadata (debugging subset).

#     Notes
#     -----
#     This function performs conversion as a side effect (writes FIF files), and
#     returns DataFrames to help users inspect what was converted vs. dropped.
#     """

#     # ------------------------------------------------------------------
#     # 0) Validate required df columns
#     # ------------------------------------------------------------------
#     required_cols = {"filepath", "subject_id", "filename"}
#     missing_cols = required_cols - set(df.columns)
#     if missing_cols:
#         raise ValueError(
#             f"[convert_butler_from_scan] df is missing required columns: {sorted(missing_cols)}"
#         )



#     # ------------------------------------------------------------------
#     # 1) Load metadata and build (subject_id_key -> label) mapping
#     # ------------------------------------------------------------------
#     metadata_path = Path(metadata_path)
#     if not metadata_path.exists():
#         raise FileNotFoundError(
#             f"[convert_butler_from_scan] metadata_path not found: {metadata_path}"
#         )

#     # Excel vs CSV
#     if metadata_path.suffix.lower() in {".xlsx", ".xls"}:
#         meta = pd.read_excel(metadata_path)
#     else:
#         meta = pd.read_csv(metadata_path)

#     if id_col not in meta.columns or label_col not in meta.columns:
#         raise ValueError(
#             f"[convert_butler_from_scan] metadata file must contain columns "
#             f"'{id_col}' and '{label_col}'. Found: {list(meta.columns)}"
#         )

#     meta = meta[[id_col, label_col]].copy()

#     # Build canonical metadata join key
#     if normalize_numeric_subject_ids:
#         meta["subject_id_key"] = meta[id_col].map(_normalize_subject_id)
#     else:
#         meta["subject_id_key"] = meta[id_col].astype(str).str.strip()

#     meta["label_meta"] = meta[label_col]

#     # Drop unusable IDs / labels
#     meta = meta.dropna(subset=["subject_id_key"])
#     meta["label_meta"] = meta["label_meta"].apply(lambda x: np.nan if pd.isna(x) else x)
#     meta = meta.dropna(subset=["label_meta"])
#     meta = meta[meta["label_meta"].astype(str).str.strip() != ""]

#     # Ensure metadata key is unique (required for validate="many_to_one")
#     meta = meta.drop_duplicates(subset=["subject_id_key"], keep="first")

#     # Optional: warn about collisions in numeric normalization
#     if normalize_numeric_subject_ids:
#         n_raw_per_key = meta.groupby("subject_id_key")[id_col].nunique()
#         if (n_raw_per_key > 1).any():
#             example = n_raw_per_key[n_raw_per_key > 1].sort_values(ascending=False).head(10)
#             print("[convert_butler_from_scan] Warning: numeric normalization caused ID collisions.")
#             print(example)

#     # ------------------------------------------------------------------
#     # 2) Build df join key and filter to subjects present in metadata
#     # ------------------------------------------------------------------
#     work = df.copy()

#     if normalize_numeric_subject_ids:
#         work["subject_id_key"] = work["subject_id"].map(_normalize_subject_id)
#     else:
#         # Resolve decorated file IDs to canonical metadata IDs (boundary-aware)
#         meta_ids = meta["subject_id_key"].dropna().astype(str).unique().tolist()
#         work["subject_id_key"] = (
#             work["subject_id"].astype(str).str.strip()
#             .map(lambda s: resolve_to_metadata_id(s, meta_ids))
#         )

#     # Unmatched subset (for user debugging)
#     uunmatched_df = work.loc[
#         work["subject_id_key"].isna() | ~work["subject_id_key"].isin(meta["subject_id_key"]),
#         ["subject_id", "subject_id_key", "filename", "filepath"]
#     ].copy()

#     # Inner join => only keep rows present in metadata
#     converted_df = work.merge(
#         meta[["subject_id_key", "label_meta"]],
#         on="subject_id_key",
#         how="inner",
#         validate="many_to_one",
#     )

#     # Metadata is source of truth for label
#     converted_df["label"] = converted_df["label_meta"]
#     converted_df = converted_df.drop(columns=["label_meta"])

#     n_in = len(df)
#     n_keep = len(converted_df)
#     n_drop = n_in - n_keep
#     print(f"[convert_butler_from_scan] Keeping {n_keep}/{n_in} files after metadata filter. Dropped {n_drop}.")

#     if strict and n_keep == 0:
#         raise ValueError("[convert_butler_from_scan] After metadata filtering, no rows remain to convert.")

#     # ------------------------------------------------------------------
#     # 3) Prepare output directory and montage
#     # ------------------------------------------------------------------
#     out_dir_path = Path(out_dir).expanduser().resolve()
#     out_dir_path.mkdir(parents=True, exist_ok=True)

#     try:
#         montage = mne.channels.make_standard_montage(montage_name)
#     except Exception as exc:
#         raise ValueError(
#             f"[convert_butler_from_scan] Invalid montage '{montage_name}'."
#         ) from exc

#     # ------------------------------------------------------------------
#     # 4) Conversion loop (write FIFs)
#     # ------------------------------------------------------------------
#     failed: list[tuple[str, str]] = []

#     for row in tqdm(converted_df.itertuples(index=False), total=len(converted_df), desc="Converting EEG → FIF (by label)", unit="file"):
#         file_path = Path(row.filepath)
#         filename = str(row.filename)
#         label = str(row.label).strip()

#         label_dir = out_dir_path / label
#         label_dir.mkdir(parents=True, exist_ok=True)

#         stem = Path(filename).stem
#         out_path = label_dir / f"{stem}.fif"

#         if out_path.exists() and not overwrite:
#             continue

#         try:
#             raw = load_raw_eeg(file_path, preload=preload)

#             if file_path.suffix.lower() == ".mat":
#                 raw = _rebuild_fieldtrip_raw_with_clean_info(raw, montage)
#             else:
#                 montage_chs = set(montage.ch_names)
#                 unknown = [ch for ch in raw.ch_names if ch not in montage_chs]
#                 if unknown:
#                     raw.set_channel_types({ch: "misc" for ch in unknown})
#                 raw.set_montage(montage)

#             raw.save(out_path.as_posix(), overwrite=overwrite, verbose=False)


#             # Keep channels not in montage (type them as misc)
#             #montage_chs = set(montage.ch_names)
#             #unknown = [ch for ch in raw.ch_names if ch not in montage_chs]
#             #if unknown:
#             #    raw.set_channel_types({ch: "misc" for ch in unknown})
#             #raw.set_montage(montage)
#             #raw.save(out_path.as_posix(), overwrite=overwrite, verbose=False)

#         except Exception as e:
#             failed.append((file_path.name, str(e)))

#     # ------------------------------------------------------------------
#     # 5) Summary
#     # ------------------------------------------------------------------
#     if failed:
#         print("\n[convert_butler_from_scan] Failed to process the following files:")
#         for fname, err in failed:
#             print(f" - {fname}: {err}")

#     return converted_df, uunmatched_df


def convert_abcct_from_scan(
    df: pd.DataFrame,
    out_dir: str | Path = "abcct_data_prepared",
    montage_name: str | None = None,
    overwrite: bool = False,
    n_blocks_to_keep: int = 3,
    require_n_blocks: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert ABC-CT *_r.mat resting EEG files (previously discovered via
    `scan_eeg_directory`) into MNE FIF format, organizing output by label.

    This function assumes that `df` already contains one row per MATLAB file,
    with at least the following columns:

        - 'filepath'   : full path to the *_r.mat file
        - 'subject_id' : subject identifier extracted from filename
        - 'label'      : label (e.g. ASD, TD, UNLABELED) [optional]

    Processing steps:
        1. Loads each .mat file using h5py (EEG_Resting + samplingRate).
        2. Selects resting segments (blocks) according to `n_blocks_to_keep` and
           `require_n_blocks`:
              - If require_n_blocks=True: skip files with < n_blocks_to_keep blocks.
              - If require_n_blocks=False: keep up to n_blocks_to_keep blocks.
        3. Converts each kept segment to an MNE Raw object.
        4. Concatenates segments, applies a standard montage.
        5. Saves one FIF per subject into label-based subfolders.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by `scan_eeg_directory` with backend="h5"
        (or an equivalent subset). Must contain 'filepath' and 'subject_id',
        and ideally 'label'.

    out_dir : str or Path
        Root output directory. Subfolders will be created per label
        (e.g., ASD, TD, UNLABELED).

    montage_name : str, optional
        Name of an MNE standard montage (e.g. "GSN-HydroCel-128").
        Must be provided and valid.

    overwrite : bool
        Whether to overwrite existing FIF files if they already exist.

    n_blocks_to_keep : int
        Target number of resting segments (blocks) to retain from EEG_Resting.

        - If require_n_blocks=True, this is a minimum requirement: files with fewer
          than `n_blocks_to_keep` blocks are skipped.
        - If require_n_blocks=False, this is a maximum: files with fewer blocks are
          still processed using all available blocks.

    require_n_blocks : bool
        If True, enforce that each file must contain at least `n_blocks_to_keep`
        blocks; otherwise skip the file (and report it). If False, process files
        even when fewer than `n_blocks_to_keep` blocks exist.
        
    Returns
    -------
    converted_df : pd.DataFrame
        Subset of input rows that were successfully converted (FIF written).
        Includes additional columns such as:
            - label (filled)
            - out_path
            - convert_status ("converted")

    uunmatched_df : pd.DataFrame
        Subset of input rows that were NOT converted for any reason.
        Includes:
            - convert_status (e.g., "skipped_exists", "skipped_insufficient_blocks", "failed")
            - convert_reason / convert_error (when applicable)

    Notes
    -----
    This function does not do folder scanning or metadata merging; that is handled
    upstream by `scan_eeg_directory`.
    """

    # ------------------------------------------------------------------
    # 0) Basic validation of df and required columns
    # ------------------------------------------------------------------
    required_cols = {"filepath", "subject_id"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[convert_abcct_from_scan] df is missing required columns: {sorted(missing_cols)}"
        )

    work = df.copy()

    # Ensure label exists and is usable
    if "label" not in work.columns:
        work["label"] = "UNLABELED"
    work["label"] = work["label"].astype("string")
    work["label"] = work["label"].fillna("").astype(str).str.strip()
    work.loc[work["label"] == "", "label"] = "UNLABELED"

    # Bookkeeping columns (so we can return converted + unmatched)
    work["out_path"] = pd.Series([pd.NA] * len(work), dtype="string")
    work["convert_status"] = pd.Series([pd.NA] * len(work), dtype="string")
    work["convert_reason"] = pd.Series([pd.NA] * len(work), dtype="string")
    work["convert_error"] = pd.Series([pd.NA] * len(work), dtype="string")

    work["n_blocks_found"] = pd.Series([pd.NA] * len(work), dtype="Int64")
    work["n_blocks_used"]  = pd.Series([pd.NA] * len(work), dtype="Int64")


    # ------------------------------------------------------------------
    # 1) Prepare output directory and montage
    # ------------------------------------------------------------------
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if montage_name is None:
        raise ValueError(
            "[convert_abcct_from_scan] You must provide montage_name "
            "(e.g., 'GSN-HydroCel-128')."
        )

    try:
        montage = mne.channels.make_standard_montage(montage_name)
    except Exception as exc:
        raise ValueError(
            f"[convert_abcct_from_scan] Invalid montage '{montage_name}'. "
            f"Check available standard montages in MNE."
        ) from exc

    # ------------------------------------------------------------------
    # 2) Track edge cases for reporting
    # ------------------------------------------------------------------
    failed: list[tuple[str, str]] = []
    insufficient_blocks: list[tuple[str, int]] = []  # (filename, n_found)
    missing_label_files: list[str] = []

    # ------------------------------------------------------------------
    # 3) Main conversion loop
    # ------------------------------------------------------------------
    for row in tqdm(
        work.itertuples(), total=len(work),
        desc="Converting EEG.mat → FIF (by label)", unit="file"
    ):
        idx = row.Index
        mat_path = Path(row.filepath)
        subj_id = str(row.subject_id)

        # label already normalized in work; keep a local copy
        label = str(row.label).strip()
        if label == "UNLABELED" and ("label" in df.columns) and (pd.isna(getattr(row, "label", np.nan))):
            missing_label_files.append(mat_path.name)

        label_dir = out_dir_path / label
        label_dir.mkdir(parents=True, exist_ok=True)

        out_path = label_dir / f"{subj_id}_eeg.fif"
        work.at[idx, "out_path"] = str(out_path)

        # Skip if output exists and we don't overwrite
        if out_path.exists() and not overwrite:
            work.at[idx, "convert_status"] = "skipped_exists"
            work.at[idx, "convert_reason"] = "Output exists and overwrite=False"
            continue

        try:
            # -----------------------------------------------------
            # 3a) Load MATLAB data with h5py
            # -----------------------------------------------------
            with h5py.File(mat_path, "r") as f:
                X = f["EEG_Resting"][()]      # (segments, time, channels)
                sr = f["samplingRate"][()]    # scalar or 1×1

            sfreq = float(sr[0, 0]) if getattr(sr, "shape", None) == (1, 1) else float(sr)

            if X.ndim != 3:
                raise ValueError(f"Unexpected EEG_Resting ndim={X.ndim} for file {mat_path.name}")

            n_seg, n_times, n_ch = X.shape
            work.at[idx, "n_blocks_found"] = int(n_seg)

            # -----------------------------------------------------
            # 3b) Decide whether to skip based on available blocks
            # -----------------------------------------------------
            if n_seg < int(n_blocks_to_keep):
                insufficient_blocks.append((mat_path.name, int(n_seg)))
                if require_n_blocks:
                    work.at[idx, "convert_status"] = "skipped_insufficient_blocks"
                    work.at[idx, "convert_reason"] = f"Found {n_seg} blocks, require >= {n_blocks_to_keep}"
                    continue

            n_keep = int(n_blocks_to_keep) if require_n_blocks else min(n_seg, int(n_blocks_to_keep))
            work.at[idx, "n_blocks_used"] = int(n_keep)
            X = X[:n_keep]

            # -----------------------------------------------------
            # 3c) Build MNE Raw objects per block
            # -----------------------------------------------------
            ch_names = [f"EEG{c+1:03d}" for c in range(n_ch)]
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

            raws: list[mne.io.BaseRaw] = []
            for i in range(n_keep):
                seg = X[i]

                # Ensure (n_channels, n_times)
                if seg.shape == (n_ch, n_times):
                    data = seg
                elif seg.shape == (n_times, n_ch):
                    data = seg.T
                else:
                    if n_ch in seg.shape:
                        ch_axis = int(np.argmax([dim == n_ch for dim in seg.shape]))
                        data = np.moveaxis(seg, ch_axis, 0)
                    else:
                        raise ValueError(
                            f"Cannot infer channel axis for segment {i} in file {mat_path.name}"
                        )

                raw_i = mne.io.RawArray(data, info.copy(), verbose=False)

                duration = data.shape[1] / sfreq
                raw_i.set_annotations(
                    mne.Annotations(
                        onset=[0.0],
                        duration=[duration],
                        description=[f"rest_block_{i+1}"],
                    )
                )
                raws.append(raw_i)

            # -----------------------------------------------------
            # 3d) Concatenate blocks + montage
            # -----------------------------------------------------
            raw = mne.concatenate_raws(raws, on_mismatch="ignore", verbose=False)

            mapping = {f"EEG{c+1:03d}": f"E{c+1}" for c in range(n_ch)}
            raw.rename_channels(mapping)

            raw.set_montage(montage)

            # -----------------------------------------------------
            # 3e) Save FIF
            # -----------------------------------------------------
            raw.save(out_path.as_posix(), overwrite=overwrite, verbose=False)

            work.at[idx, "convert_status"] = "converted"
            work.at[idx, "convert_reason"] = "OK"

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            failed.append((mat_path.name, msg))

            work.at[idx, "convert_status"] = "failed"
            work.at[idx, "convert_reason"] = "Exception during conversion"
            work.at[idx, "convert_error"] = msg

    # ------------------------------------------------------------------
    # 4) Summary reporting
    # ------------------------------------------------------------------
    converted_df = work.loc[work["convert_status"] == "converted"].copy()
    uunmatched_df = work.loc[work["convert_status"] != "converted"].copy()

    print(f"[convert_abcct_from_scan] Converted {len(converted_df)}/{len(work)} files.")
    print(uunmatched_df["convert_status"].value_counts(dropna=False))


    return converted_df, uunmatched_df


def _normalize_epochs_input(
    epochs_or_X,
    sfreq: Optional[float] = None,
    ch_names: Optional[list[str]] = None,
):
    """
    Internal helper — normalizes input to always return (X, sfreq, ch_names).
    Supports both MNE Epochs and NumPy arrays.

    Parameters
    ----------
    epochs_or_X : mne.Epochs | np.ndarray
        EEG data source, either an MNE Epochs object or a NumPy array.
    sfreq : float, optional
        Sampling frequency in Hz. Required when using NumPy arrays.
    ch_names : list of str, optional
        Channel names. Required when using NumPy arrays.

    Returns
    -------
    X : np.ndarray
        EEG data as array (n_epochs, n_channels, n_times)
    sf : float
        Sampling frequency (Hz)
    chn : list of str
        Channel names
    """
    try:
        import mne
        is_mne = isinstance(epochs_or_X, mne.Epochs)
    except Exception:
        is_mne = False

    if is_mne:
        # MNE object → extract data, sfreq, and ch_names
        X = epochs_or_X.get_data()
        sf = float(epochs_or_X.info["sfreq"])
        chn = list(epochs_or_X.ch_names)
    else:
        # Numpy array path → must supply sfreq and ch_names
        X = np.asarray(epochs_or_X)
        if X.ndim != 3:
            raise ValueError("Expected ndarray with shape (n_epochs, n_channels, n_times).")
        if sfreq is None:
            raise ValueError("When passing a numpy array, you must provide `sfreq`.")
        if ch_names is None:
            raise ValueError("When passing a numpy array, you must provide `ch_names` (list[str]).")
        if len(ch_names) != X.shape[1]:
            raise ValueError(
                f"ch_names length ({len(ch_names)}) must match n_channels ({X.shape[1]})."
            )
        sf = float(sfreq)
        chn = ch_names

    # Robust guard: replace NaN/Inf with finite values in-place
    np.nan_to_num(X, copy=False)
    return X, sf, chn



def _resolve_final_epochs_output(
    state: Mapping[str, Any],
    final_epochs_key: str = "epochs_clean",
) -> tuple[mne.BaseEpochs | dict[str, mne.BaseEpochs], bool]:
    """Resolve single-recording or condition-specific final Epochs from pipeline state."""
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping.")
    if not final_epochs_key:
        raise ValueError("final_epochs_key cannot be empty.")

    # Prefer condition-specific output when available.
    condition_key = f"{final_epochs_key}_by_condition"
    if condition_key in state:
        epochs_by_condition = state[condition_key]
        if not isinstance(epochs_by_condition, Mapping) or not epochs_by_condition:
            raise TypeError(f"state['{condition_key}'] must be a non-empty mapping.")

        resolved: dict[str, mne.BaseEpochs] = {}
        for condition, epochs in epochs_by_condition.items():
            condition = str(condition).strip()
            if not condition:
                raise ValueError(f"state['{condition_key}'] contains an empty condition name.")
            if not isinstance(epochs, mne.BaseEpochs):
                raise TypeError(
                    f"state['{condition_key}']['{condition}'] must be an MNE Epochs object, "
                    f"received {type(epochs).__name__}."
                )
            if len(epochs) == 0:
                raise ValueError(f"state['{condition_key}']['{condition}'] contains no retained epochs.")
            resolved[condition] = epochs.copy()

        return resolved, True

    # Backward-compatible single-recording path.
    if final_epochs_key in state:
        epochs = state[final_epochs_key]
        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError(
                f"state['{final_epochs_key}'] must be an MNE Epochs object, "
                f"received {type(epochs).__name__}."
            )
        if len(epochs) == 0:
            raise ValueError(f"state['{final_epochs_key}'] contains no retained epochs.")
        return epochs.copy(), False

    raise KeyError(
        f"Pipeline state contains neither '{final_epochs_key}' nor "
        f"'{condition_key}'. Available keys: {sorted(state.keys())}"
    )


def _build_condition_epoch_qc(
    state: Mapping[str, Any],
    condition: str,
    epochs: mne.BaseEpochs,
    *,
    condition_to_eye_state: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build condition-specific epoch/QC metrics from preprocessing output."""
    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("epochs must be an MNE Epochs object.")

    condition = str(condition)
    if condition_to_eye_state is None:
        condition_to_eye_state = {"EO": "EO", "EC": "EC"}

    sfreq = float(epochs.info["sfreq"])
    n_retained = len(epochs)
    n_samples = len(epochs.times)
    epoch_duration = n_samples / sfreq

    # Prefer the rejection-step summary because it contains exact pre/post counts.
    summaries = state.get("bad_epoch_rejection_summary_by_condition", {})
    summary = summaries.get(condition) if isinstance(summaries, Mapping) else None

    if isinstance(summary, Mapping):
        n_attempted = int(summary["n_epochs_before"])
        n_rejected = int(summary["n_epochs_rejected"])
        n_retained = int(summary["n_epochs_retained"])
        rejection_threshold = summary.get("reject_thresholds")
    else:
        # Fallback when condition-specific rejection was not run.
        drop_log = list(epochs.drop_log)
        n_ignored = sum(bool(reasons) and all(r == "IGNORED" for r in reasons) for reasons in drop_log)
        n_attempted = max(len(drop_log) - n_ignored, n_retained)
        n_rejected = max(n_attempted - n_retained, 0)

        thresholds = state.get("bad_epoch_reject_thresholds_by_condition", {})
        rejection_threshold = thresholds.get(condition) if isinstance(thresholds, Mapping) else None

    retention = 100.0 * n_retained / n_attempted if n_attempted else 0.0
    usable_seconds = n_retained * epoch_duration

    # Capture actual rejection reasons for this condition.
    reasons_by_condition = state.get("bad_epoch_drop_reasons_by_condition", {})
    condition_reasons = reasons_by_condition.get(condition, {}) if isinstance(reasons_by_condition, Mapping) else {}
    drop_reason_counts = Counter(
        reason
        for reasons in condition_reasons.values()
        for reason in reasons
        if reason != "IGNORED"
    )

    eeg_picks = mne.pick_types(
        epochs.info,
        eeg=True,
        meg=False,
        stim=False,
        misc=False,
        exclude=[],
    )

    qc = {
        "analysis_condition": condition,
        "final_sfreq_hz": sfreq,
        "final_n_channels": len(epochs.ch_names),
        "final_n_eeg_channels": len(eeg_picks),
        "n_samples_per_epoch": n_samples,
        "epoch_duration_seconds": epoch_duration,
        "n_epochs_attempted": n_attempted,
        "n_epochs_rejected": n_rejected,
        "n_epochs_retained": n_retained,
        "epoch_retention_percent": retention,
        "usable_clean_seconds": usable_seconds,
        "usable_clean_minutes": usable_seconds / 60.0,
        "epoch_drop_reason_counts": dict(drop_reason_counts),
        "rejection_threshold": rejection_threshold,
    }

    # Only populate eye_state when the condition actually represents an eye state.
    if condition in condition_to_eye_state:
        qc["eye_state"] = str(condition_to_eye_state[condition])

    return qc



# def build_label_epoch_arrays(
#     data_root: str | Path,
#     base_config: dict[str, Any],
#     skip_dirnames: Sequence[str] | None = None,
#     final_epochs_key: str = "epochs_clean",
#     verbose: bool = True,
# ) -> tuple[
#     dict[str, list[mne.BaseEpochs]],
#     list[dict[str, Any]],
#     dict[str, Any],
#     list[dict[str, Any]],
# ]:
#     """
#     Preprocess labeled EEG files and preserve recording-level QC information.

#     The expected directory structure is:

#         data_root/
#             ASD/
#                 subject_1_eeg.fif
#                 subject_2_eeg.fif
#             TD/
#                 subject_3_eeg.fif

#     Each label folder is processed independently. Successfully processed
#     recordings are stored as MNE Epochs objects, while a compact QC record is
#     created for every attempted file, including files that fail preprocessing.

#     Parameters
#     ----------
#     data_root
#         Root directory containing one subdirectory per label.

#     base_config
#         Base configuration passed to ``eeg_preprocess_pipeline``. The path in
#         the ``load_eeg`` step is replaced with the current file path.

#     skip_dirnames
#         Label-directory names to exclude. Matching is case-insensitive.
#         Defaults to ``("UNLABELED",)``.

#     final_epochs_key
#         Pipeline-state key containing the final cleaned Epochs object.
#         Defaults to ``"epochs_clean"``.

#     verbose
#         Whether to print batch-level progress and completion summaries.

#     Returns
#     -------
#     results
#         Dictionary mapping each label to its successfully processed MNE
#         Epochs objects.

#     metadata
#         One dictionary per successfully processed recording containing:

#         - ``file_path``
#         - ``recording_id``
#         - ``label``
#         - ``subject_id``
#         - ``label_idx``
#         - ``global_idx``
#         - ``qc_idx``

#     eeg_info
#         Sampling rate and channel names from the first successful recording.

#     qc_records
#         One QC dictionary for every attempted recording, including failures.

#         Important fields include:

#         - processing status and error
#         - input recording duration and channel count
#         - retained, rejected, and attempted epoch counts
#         - epoch-retention percentage
#         - usable clean minutes
#         - RANSAC bad/interpolated channels
#         - ICA components fitted and excluded
#         - demoted unlocalized channels
#         - epoch-drop reasons
#         - basic QC flag and notes

#     Notes
#     -----
#     - Failed recordings are included in ``qc_records`` but are not added to
#       ``results`` or ``metadata``.
#     - ``label_idx`` and ``global_idx`` count successful recordings only.
#     - ``discovery_idx`` in the QC output counts every attempted file.
#     - Epoch rejection is derived from the final Epochs ``drop_log`` so the
#       function does not depend on a particular rejection-step state key.
#     """
#     # ------------------------------------------------------------------
#     # Validate inputs
#     # ------------------------------------------------------------------
#     data_root = Path(data_root)

#     if not data_root.exists() or not data_root.is_dir():
#         raise ValueError(
#             f"Data root '{data_root}' does not exist or is not a directory."
#         )

#     if not isinstance(base_config, dict):
#         raise TypeError("base_config must be a dictionary.")

#     if not isinstance(base_config.get("steps"), list):
#         raise ValueError("base_config must contain a list under 'steps'.")

#     if not final_epochs_key:
#         raise ValueError("final_epochs_key cannot be empty.")



#     # Use a nonmutable default and compare skipped directory names
#     # without regard to capitalization.
#     skip_dirnames = ("UNLABELED",) if skip_dirnames is None else skip_dirnames
#     skip_set = {str(name).lower() for name in skip_dirnames}

#     # ------------------------------------------------------------------
#     # Discover label folders and FIF recordings
#     # ------------------------------------------------------------------
#     label_to_files: dict[str, list[Path]] = {}

#     for subdirectory in sorted(data_root.iterdir()):
#         if not subdirectory.is_dir():
#             continue
#         if subdirectory.name.lower() in skip_set:
#             continue

#         fif_files = sorted(subdirectory.glob("*.fif"))

#         if fif_files:
#             label_to_files[subdirectory.name] = fif_files

#     if not label_to_files:
#         raise ValueError(
#             f"No label folders containing .fif files were found under "
#             f"'{data_root}'."
#         )

#     n_discovered = sum(len(files) for files in label_to_files.values())

#     if verbose:
#         print(
#             f"Discovered {n_discovered} EEG recordings across "
#             f"{len(label_to_files)} label folders."
#         )

#     # ------------------------------------------------------------------
#     # Initialize outputs and deterministic counters
#     # ------------------------------------------------------------------
#     results: dict[str, list[mne.BaseEpochs]] = defaultdict(list)
#     metadata: list[dict[str, Any]] = []
#     qc_records: list[dict[str, Any]] = []
#     eeg_info: dict[str, Any] = {}

#     per_label_counts: dict[str, int] = defaultdict(int)
#     global_count = 0
#     discovery_idx = 0

#     # ------------------------------------------------------------------
#     # Preprocess every recording independently
#     # ------------------------------------------------------------------
#     for label, files in label_to_files.items():
#         for fpath in files:
#             processing_start = perf_counter()

#             recording_id = fpath.stem
#             subject_id = recording_id.removesuffix("_eeg")

#             # Initialize the failure-safe QC record before preprocessing.
#             qc_record: dict[str, Any] = {
#                 "discovery_idx": discovery_idx,
#                 "file_path": str(fpath),
#                 "recording_id": recording_id,
#                 "label": label,
#                 "subject_id": subject_id,
#                 "processing_status": "failed",
#                 "processing_error": None,
#                 "final_epochs_key": final_epochs_key,

#                 "input_sfreq_hz": None,
#                 "input_n_channels": None,
#                 "input_n_eeg_channels": None,
#                 "input_n_times": None,
#                 "input_duration_seconds": None,

#                 "final_sfreq_hz": None,
#                 "final_n_channels": None,
#                 "final_n_eeg_channels": None,
#                 "final_raw_duration_seconds": None,
#                 "n_samples_per_epoch": None,
#                 "epoch_duration_seconds": None,

#                 "n_epochs_attempted": None,
#                 "n_epochs_rejected": None,
#                 "n_epochs_retained": None,
#                 "epoch_retention_percent": None,
#                 "usable_clean_seconds": None,
#                 "usable_clean_minutes": None,


#                 "bad_channels": [],
#                 "n_bad_channels": 0,
#                 "demoted_unlocalized": [],
#                 "n_demoted_unlocalized": 0,

#                 "ica_n_components": None,
#                 "excluded_ics": [],
#                 "n_excluded_ics": 0,
#                 "excluded_ic_labels": [],

#                 "epoch_drop_reason_counts": {},
#                 "rejection_threshold_state_key": None,
#                 "rejection_threshold": None,

#                 "scale_factor": None,
#                 "csd_applied": False,

#                 "sfreq_matches_reference": None,
#                 "channels_match_reference": None,

#                 "qc_flag": "failed",
#                 "qc_notes": [],
#                 "processing_seconds": None,
#             }

#             discovery_idx += 1

#             try:
#                 # ------------------------------------------------------
#                 # Read lightweight information from the prepared FIF file
#                 # before the preprocessing pipeline modifies it.
#                 # ------------------------------------------------------
#                 input_raw = mne.io.read_raw_fif(
#                     fpath,
#                     preload=False,
#                     verbose=False,
#                 )

#                 try:
#                     input_sfreq = float(input_raw.info["sfreq"])
#                     input_n_times = int(input_raw.n_times)

#                     input_eeg_picks = mne.pick_types(
#                         input_raw.info,
#                         eeg=True,
#                         meg=False,
#                         stim=False,
#                         misc=False,
#                         exclude=[],
#                     )

#                     qc_record.update({
#                         "input_sfreq_hz": input_sfreq,
#                         "input_n_channels": len(input_raw.ch_names),
#                         "input_n_eeg_channels": len(input_eeg_picks),
#                         "input_n_times": input_n_times,
#                         "input_duration_seconds": (
#                             input_n_times / input_sfreq
#                         ),
#                     })
#                 finally:
#                     input_raw.close()

#                 # ------------------------------------------------------
#                 # Copy the shared configuration and replace load_eeg path
#                 # ------------------------------------------------------
#                 cfg = deepcopy(base_config)
#                 load_step_found = False

#                 for step in cfg["steps"]:
#                     if "load_eeg" not in step:
#                         continue

#                     load_spec = step["load_eeg"]

#                     if load_spec is None:
#                         step["load_eeg"] = {
#                             "params": {"path": str(fpath)},
#                             "verbose": False,
#                         }
#                     elif "params" in load_spec:
#                         load_spec.setdefault("params", {})
#                         load_spec["params"]["path"] = str(fpath)
#                     else:
#                         load_spec["path"] = str(fpath)

#                     load_step_found = True
#                     break

#                 if not load_step_found:
#                     raise KeyError(
#                         "base_config does not contain a 'load_eeg' step."
#                     )

#                 # ------------------------------------------------------
#                 # Run the complete preprocessing pipeline
#                 # ------------------------------------------------------

#                 state = eeg_preprocess_pipeline(cfg)

#                 if final_epochs_key not in state:
#                     raise KeyError(
#                         f"Pipeline state does not contain "
#                         f"'{final_epochs_key}'. Available keys: "
#                         f"{sorted(state.keys())}"
#                     )

#                 epochs_final = state[final_epochs_key].copy()

#                 if not isinstance(epochs_final, mne.BaseEpochs):
#                     raise TypeError(
#                         f"state['{final_epochs_key}'] must be an MNE "
#                         f"Epochs object, received "
#                         f"{type(epochs_final).__name__}."
#                     )

#                 if len(epochs_final) == 0:
#                     raise ValueError(
#                         f"state['{final_epochs_key}'] contains no "
#                         "retained epochs."
#                     )

#                 # ------------------------------------------------------
#                 # Derive final epoch and usable-duration QC
#                 # ------------------------------------------------------
#                 final_sfreq = float(epochs_final.info["sfreq"])
#                 n_retained = int(len(epochs_final))
#                 n_samples_per_epoch = int(len(epochs_final.times))

#                 # Using samples / sfreq gives exactly 2.0 seconds for
#                 # 500 samples at 250 Hz, even when tmax is 1.996 seconds.
#                 epoch_duration_seconds = (
#                     n_samples_per_epoch / final_sfreq
#                 )

#                 drop_log = tuple(
#                     getattr(epochs_final, "drop_log", ())
#                 )

#                 # MNE may include events marked IGNORED in drop_log.
#                 # These were not candidate epochs and should not count
#                 # as rejected analysis epochs.
#                 n_ignored = sum(
#                     "IGNORED" in reasons
#                     for reasons in drop_log
#                 )

#                 n_attempted = (
#                     len(drop_log) - n_ignored
#                     if drop_log
#                     else n_retained
#                 )

#                 n_attempted = max(n_attempted, n_retained)
#                 n_rejected = max(n_attempted - n_retained, 0)

#                 retention_percent = (
#                     100.0 * n_retained / n_attempted
#                     if n_attempted > 0
#                     else 0.0
#                 )

#                 usable_clean_seconds = (
#                     n_retained * epoch_duration_seconds
#                 )
#                 usable_clean_minutes = usable_clean_seconds / 60.0

#                 # Count each MNE epoch-drop reason for later QC summaries.
#                 drop_reason_counts = Counter(
#                     reason
#                     for reasons in drop_log
#                     for reason in reasons
#                     if reason != "IGNORED"
#                 )

#                 final_eeg_picks = mne.pick_types(
#                     epochs_final.info,
#                     eeg=True,
#                     meg=False,
#                     stim=False,
#                     misc=False,
#                     exclude=[],
#                 )

#                 # ------------------------------------------------------
#                 # Capture RANSAC, localization, and ICA information
#                 # ------------------------------------------------------
#                 bad_channels = [
#                     str(channel)
#                     for channel in state.get("bad_channels", [])
#                 ]

#                 demoted_unlocalized = [
#                     str(channel)
#                     for channel in state.get(
#                         "demoted_unlocalized",
#                         [],
#                     )
#                 ]

#                 excluded_ics = [
#                     int(component)
#                     for component in state.get("excluded_ics", [])
#                 ]

#                 ica = state.get("ica")
#                 ica_n_components = getattr(
#                     ica,
#                     "n_components_",
#                     None,
#                 )

#                 if ica_n_components is not None:
#                     ica_n_components = int(ica_n_components)

#                 # Extract the ICLabel classes corresponding to removed ICs.
#                 excluded_ic_labels: list[str] = []
#                 iclabel_df = state.get("iclabel_df")

#                 if (
#                     isinstance(iclabel_df, pd.DataFrame)
#                     and {"label", "excluded"}.issubset(iclabel_df.columns)
#                 ):
#                     excluded_ic_labels = (
#                         iclabel_df.loc[
#                             iclabel_df["excluded"].astype(bool),
#                             "label",
#                         ]
#                         .astype(str)
#                         .tolist()
#                     )

#                 # ------------------------------------------------------
#                 # Capture an epoch-rejection threshold when the pipeline
#                 # stores one. The source key is retained for traceability.
#                 # ------------------------------------------------------
#                 threshold_key = next(
#                     (
#                         key
#                         for key in (
#                             "rejection_threshold",
#                             "reject_threshold",
#                             "rejection_thresholds",
#                             "reject_thresholds",
#                         )
#                         if key in state
#                     ),
#                     None,
#                 )

#                 rejection_threshold = (
#                     state[threshold_key]
#                     if threshold_key is not None
#                     else None
#                 )

#                 # ------------------------------------------------------
#                 # Capture final Raw information when it remains in state
#                 # ------------------------------------------------------
#                 final_raw = state.get("raw")

#                 if isinstance(final_raw, mne.io.BaseRaw):
#                     final_raw_duration_seconds = (
#                         final_raw.n_times
#                         / float(final_raw.info["sfreq"])
#                     )
#                 else:
#                     final_raw_duration_seconds = None

#                 # ------------------------------------------------------
#                 # Establish or compare against the reference EEG layout
#                 # ------------------------------------------------------
#                 if not eeg_info:
#                     eeg_info = {
#                         "sfreq": final_sfreq,
#                         "ch_names": list(epochs_final.ch_names),
#                     }

#                     sfreq_matches_reference = True
#                     channels_match_reference = True
#                 else:
#                     sfreq_matches_reference = (
#                         abs(final_sfreq - eeg_info["sfreq"]) <= 1e-6
#                     )

#                     channels_match_reference = (
#                         list(epochs_final.ch_names)
#                         == eeg_info["ch_names"]
#                     )

#                     if not sfreq_matches_reference:
#                         print(
#                             f"[warn] sfreq mismatch in '{fpath}': "
#                             f"got {final_sfreq}, "
#                             f"expected {eeg_info['sfreq']}."
#                         )

#                     if not channels_match_reference:
#                         print(
#                             f"[warn] ch_names mismatch in '{fpath}'. "
#                             "Using channel names from the first success."
#                         )

#                 # ------------------------------------------------------
#                 # Assign a simple, transparent QC flag
#                 # ------------------------------------------------------

#                 qc_notes: list[str] = []

#                 if not sfreq_matches_reference:
#                     qc_notes.append(
#                         "Sampling rate differs from the first successful "
#                         "recording."
#                     )

#                 if not channels_match_reference:
#                     qc_notes.append(
#                         "Channel names or ordering differ from the first "
#                         "successful recording."
#                     )

#                 qc_flag = "pass" if not qc_notes else "review"

#                 qc_record.update({
#                     "processing_status": "success",
#                     "processing_error": None,

#                     "final_sfreq_hz": final_sfreq,
#                     "final_n_channels": len(epochs_final.ch_names),
#                     "final_n_eeg_channels": len(final_eeg_picks),
#                     "final_raw_duration_seconds": (
#                         final_raw_duration_seconds
#                     ),
#                     "n_samples_per_epoch": n_samples_per_epoch,
#                     "epoch_duration_seconds": epoch_duration_seconds,

#                     "n_epochs_attempted": n_attempted,
#                     "n_epochs_rejected": n_rejected,
#                     "n_epochs_retained": n_retained,
#                     "epoch_retention_percent": retention_percent,
#                     "usable_clean_seconds": usable_clean_seconds,
#                     "usable_clean_minutes": usable_clean_minutes,

#                     "bad_channels": bad_channels,
#                     "n_bad_channels": len(bad_channels),

#                     "demoted_unlocalized": demoted_unlocalized,
#                     "n_demoted_unlocalized": len(
#                         demoted_unlocalized
#                     ),

#                     "ica_n_components": ica_n_components,
#                     "excluded_ics": excluded_ics,
#                     "n_excluded_ics": len(excluded_ics),
#                     "excluded_ic_labels": excluded_ic_labels,

#                     "epoch_drop_reason_counts": dict(
#                         drop_reason_counts
#                     ),
#                     "rejection_threshold_state_key": threshold_key,
#                     "rejection_threshold": rejection_threshold,

#                     "scale_factor": state.get("scale_factor"),
#                     "csd_applied": bool(
#                         state.get("csd_applied", False)
#                     ),

#                     "sfreq_matches_reference": (
#                         sfreq_matches_reference
#                     ),
#                     "channels_match_reference": (
#                         channels_match_reference
#                     ),

#                     "qc_flag": qc_flag,
#                     "qc_notes": qc_notes,
#                 })

#                 # ------------------------------------------------------
#                 # Store successful Epochs and deterministic metadata
#                 # ------------------------------------------------------
#                 qc_idx = len(qc_records)

#                 results[label].append(epochs_final)

#                 metadata.append({
#                     "file_path": str(fpath),
#                     "recording_id": recording_id,
#                     "label": label,
#                     "subject_id": subject_id,
#                     "label_idx": per_label_counts[label],
#                     "global_idx": global_count,
#                     "qc_idx": qc_idx,
#                 })

#                 per_label_counts[label] += 1
#                 global_count += 1

#             except Exception as exc:
#                 # Preserve failed attempts in QC so attempted N and reasons
#                 # remain available for the study-level completeness report.
#                 qc_record["processing_error"] = (
#                     f"{type(exc).__name__}: {exc}"
#                 )
#                 qc_record["qc_notes"] = [
#                     "Preprocessing failed."
#                 ]

#                 print(
#                     f"[warn] Skipping '{fpath}' ({label}): {exc}"
#                 )

#             finally:
#                 qc_record["processing_seconds"] = (
#                     perf_counter() - processing_start
#                 )
#                 qc_records.append(qc_record)

#     if verbose:
#         n_success = sum(
#             record["processing_status"] == "success"
#             for record in qc_records
#         )
#         n_failed = len(qc_records) - n_success
#         n_review = sum(
#             record["qc_flag"] == "review"
#             for record in qc_records
#         )

#         print("\n" + "=" * 80)
#         print("Batch preprocessing summary")
#         print("=" * 80)
#         print(f"Attempted recordings: {len(qc_records)}")
#         print(f"Successful:           {n_success}")
#         print(f"Failed:               {n_failed}")
#         print(f"Flagged for review:   {n_review}")
#         print("=" * 80)

#     return dict(results), metadata, eeg_info, qc_records


def build_label_epoch_arrays(
    data_root: str | Path,
    base_config: dict[str, Any],
    skip_dirnames: Sequence[str] | None = None,
    final_epochs_key: str = "epochs_clean",
    subject_id_borders: Sequence[str] | None = None,
    timepoint_regex: str | None = None,
    condition_to_eye_state: Mapping[str, str] | None = None,
    verbose: bool = True,
) -> tuple[
    dict[str, list[mne.BaseEpochs | dict[str, mne.BaseEpochs]]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Preprocess EEG files and build analysis-ready Epochs, metadata, and QC.

    Supports both:
      1. Standard recordings -> one cleaned Epochs object per physical file.
      2. Condition-aware recordings -> one mapping of condition -> cleaned Epochs
         per physical file (e.g., EC and EO).

    The physical recording is preprocessed once. Condition-specific Epochs and QC
    are then preserved separately for downstream qEEG analysis.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    data_root = Path(data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise ValueError(f"Data root '{data_root}' does not exist or is not a directory.")
    if not isinstance(base_config, dict):
        raise TypeError("base_config must be a dictionary.")
    if not isinstance(base_config.get("steps"), list):
        raise ValueError("base_config must contain a list under 'steps'.")
    if not final_epochs_key:
        raise ValueError("final_epochs_key cannot be empty.")
    if condition_to_eye_state is not None and not isinstance(condition_to_eye_state, Mapping):
        raise TypeError("condition_to_eye_state must be a mapping or None.")

    # Compile optional study-specific timepoint parser once.
    timepoint_pattern = re.compile(timepoint_regex) if timepoint_regex else None

    # Preserve previous behavior: UNLABELED is skipped unless explicitly allowed.
    if skip_dirnames is None:
        skip_dirnames = ("UNLABELED",)
    elif isinstance(skip_dirnames, str):
        skip_dirnames = (skip_dirnames,)
    skip_set = {str(name).lower() for name in skip_dirnames}

    # ------------------------------------------------------------------
    # Discover label folders and FIF recordings
    # ------------------------------------------------------------------
    label_to_files: dict[str, list[Path]] = {}
    for subdirectory in sorted(data_root.iterdir()):
        if not subdirectory.is_dir() or subdirectory.name.lower() in skip_set:
            continue
        fif_files = sorted(subdirectory.glob("*.fif"))
        if fif_files:
            label_to_files[subdirectory.name] = fif_files

    if not label_to_files:
        raise ValueError(
            f"No label folders containing .fif files were found under '{data_root}'."
        )

    n_discovered = sum(len(files) for files in label_to_files.values())
    if verbose:
        print(
            f"Discovered {n_discovered} EEG recordings across "
            f"{len(label_to_files)} label folders."
        )

    # ------------------------------------------------------------------
    # Initialize outputs
    # ------------------------------------------------------------------
    results: dict[
        str,
        list[mne.BaseEpochs | dict[str, mne.BaseEpochs]],
    ] = defaultdict(list)
    metadata: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []
    eeg_info: dict[str, Any] = {}

    per_label_counts: dict[str, int] = defaultdict(int)
    global_count = 0
    discovery_idx = 0

    # ------------------------------------------------------------------
    # Helper: build QC for original single-Epochs workflow
    # ------------------------------------------------------------------
    def _build_single_epoch_qc(
        state: Mapping[str, Any],
        epochs: mne.BaseEpochs,
    ) -> tuple[dict[str, Any], str | None]:
        final_sfreq = float(epochs.info["sfreq"])
        n_retained = int(len(epochs))
        n_samples = int(len(epochs.times))
        epoch_duration = n_samples / final_sfreq

        # Prefer exact summary generated by reject_bad_epochs.
        summary = state.get("bad_epoch_rejection_summary")
        if isinstance(summary, Mapping):
            n_attempted = int(summary["n_epochs_before"])
            n_rejected = int(summary["n_epochs_rejected"])
            n_retained = int(summary["n_epochs_retained"])
            rejection_threshold = summary.get("reject_thresholds")
            threshold_key = (
                "bad_epoch_reject_thresholds"
                if "bad_epoch_reject_thresholds" in state
                else "bad_epoch_rejection_summary"
            )

            reasons = state.get("bad_epoch_drop_reasons", {})
            drop_reason_counts = Counter(
                reason
                for reason_list in reasons.values()
                for reason in reason_list
                if reason != "IGNORED"
            ) if isinstance(reasons, Mapping) else Counter()
        else:
            # Backward-compatible fallback using MNE drop_log.
            drop_log = tuple(getattr(epochs, "drop_log", ()))
            n_ignored = sum(
                bool(reasons) and all(reason == "IGNORED" for reason in reasons)
                for reasons in drop_log
            )
            n_attempted = max(
                len(drop_log) - n_ignored if drop_log else n_retained,
                n_retained,
            )
            n_rejected = max(n_attempted - n_retained, 0)
            drop_reason_counts = Counter(
                reason
                for reasons in drop_log
                for reason in reasons
                if reason != "IGNORED"
            )

            threshold_key = next(
                (
                    key for key in (
                        "bad_epoch_reject_thresholds",
                        "rejection_threshold",
                        "reject_threshold",
                        "rejection_thresholds",
                        "reject_thresholds",
                    )
                    if key in state
                ),
                None,
            )
            rejection_threshold = (
                state[threshold_key]
                if threshold_key is not None
                else None
            )

        eeg_picks = mne.pick_types(
            epochs.info,
            eeg=True,
            meg=False,
            stim=False,
            misc=False,
            exclude=[],
        )
        retention = 100.0 * n_retained / n_attempted if n_attempted else 0.0
        usable_seconds = n_retained * epoch_duration

        return {
            "final_sfreq_hz": final_sfreq,
            "final_n_channels": len(epochs.ch_names),
            "final_n_eeg_channels": len(eeg_picks),
            "n_samples_per_epoch": n_samples,
            "epoch_duration_seconds": epoch_duration,
            "n_epochs_attempted": n_attempted,
            "n_epochs_rejected": n_rejected,
            "n_epochs_retained": n_retained,
            "epoch_retention_percent": retention,
            "usable_clean_seconds": usable_seconds,
            "usable_clean_minutes": usable_seconds / 60.0,
            "epoch_drop_reason_counts": dict(drop_reason_counts),
            "rejection_threshold": rejection_threshold,
        }, threshold_key

    # ------------------------------------------------------------------
    # Preprocess every physical recording once
    # ------------------------------------------------------------------
    for label, files in label_to_files.items():
        for fpath in files:
            processing_start = perf_counter()
            recording_id = fpath.stem

            # Configurable subject-ID parsing; default preserves old behavior.
            subject_id = subject_id_from_borders(
                fpath,
                list(subject_id_borders) if subject_id_borders else None,
            ).removesuffix("_eeg")

            # Optional configurable timepoint extraction.
            timepoint = None
            if timepoint_pattern is not None:
                match = timepoint_pattern.search(recording_id)
                if match:
                    if "timepoint" in match.groupdict():
                        timepoint = match.group("timepoint")
                    elif match.lastindex:
                        timepoint = match.group(1)
                    else:
                        timepoint = match.group(0)

            # Failure-safe physical-record QC template.
            qc_base: dict[str, Any] = {
                "discovery_idx": discovery_idx,
                "file_path": str(fpath),
                "recording_id": recording_id,
                "source_recording_id": recording_id,
                "label": label,
                "subject_id": subject_id,
                "timepoint": timepoint,
                "analysis_condition": None,
                "eye_state": None,
                "condition_idx": None,
                "condition_mode": False,
                "label_idx": None,
                "global_idx": None,
                "qc_idx": None,
                "processing_status": "failed",
                "processing_error": None,
                "final_epochs_key": final_epochs_key,

                "input_sfreq_hz": None,
                "input_n_channels": None,
                "input_n_eeg_channels": None,
                "input_n_eog_channels": None,
                "input_eog_channels": [],
                "input_channel_types": {},
                "input_n_times": None,
                "input_duration_seconds": None,

                "final_sfreq_hz": None,
                "final_n_channels": None,
                "final_n_eeg_channels": None,
                "final_n_eog_channels": None,
                "final_eog_channels": [],
                "final_channel_types": {},
                "final_raw_duration_seconds": None,


                "n_samples_per_epoch": None,
                "epoch_duration_seconds": None,
                "n_epochs_attempted": None,
                "n_epochs_rejected": None,
                "n_epochs_retained": None,
                "epoch_retention_percent": None,
                "usable_clean_seconds": None,
                "usable_clean_minutes": None,

                # Bad-channel / spatial QC
                "mad_bad_channels": [],                 # Channels detected by MAD amplitude-outlier detection
                "n_mad_bad_channels": 0,                # Number of MAD-detected channels

                "ransac_bad_channels": [],              # Channels detected by RANSAC
                "n_ransac_bad_channels": 0,             # Number of RANSAC-detected channels

                "bad_channels": [],                     # Final union of MAD + RANSAC interpolated channels
                "n_bad_channels": 0,                    # Number of unique interpolated channels

                "demoted_unlocalized": [],
                "n_demoted_unlocalized": 0,


                "ica_n_components": None,
                "excluded_ics": [],
                "n_excluded_ics": 0,
                "excluded_ic_labels": [],

                # EOG-supported ICA QC
                "eog_available": False,
                "eog_channels": [],
                "eog_candidate_ics": [],
                "n_eog_candidate_ics": 0,
                
                "epoch_drop_reason_counts": {},
                "rejection_threshold_state_key": None,
                "rejection_threshold": None,
                "scale_factor": None,
                "csd_applied": False,
                "sfreq_matches_reference": None,
                "channels_match_reference": None,
                "qc_flag": "failed",
                "qc_notes": [],
                "processing_seconds": None,
            }
            discovery_idx += 1

            try:
                # ------------------------------------------------------
                # Read original prepared FIF information
                # ------------------------------------------------------
                input_raw = mne.io.read_raw_fif(
                    fpath,
                    preload=False,
                    verbose=False,
                )
                try:

                    input_sfreq = float(input_raw.info["sfreq"])
                    input_n_times = int(input_raw.n_times)

                    # ------------------------------------------------------
                    # Identify EEG channels entering the main preprocessing/qEEG workflow.
                    # ------------------------------------------------------
                    input_eeg_picks = mne.pick_types(
                        input_raw.info,
                        eeg=True,
                        eog=False,
                        meg=False,
                        stim=False,
                        misc=False,
                        exclude=[],
                    )

                    # ------------------------------------------------------
                    # Identify all channels explicitly typed as EOG.
                    # Channel names are not hardcoded here; any channel whose
                    # MNE type is "eog" will automatically be detected.
                    # ------------------------------------------------------
                    input_eog_picks = mne.pick_types(
                        input_raw.info,
                        eeg=False,
                        eog=True,
                        meg=False,
                        stim=False,
                        misc=False,
                        exclude=[],
                    )

                    input_eog_channels = [
                        input_raw.ch_names[idx]
                        for idx in input_eog_picks
                    ]

                    # Preserve the complete channel-name -> channel-type mapping
                    # for QC, traceability, and future datasets containing other
                    # auxiliary channel types.
                    input_channel_types = dict(
                        zip(
                            input_raw.ch_names,
                            input_raw.get_channel_types(),
                        )
                    )

                    qc_base.update({
                        "input_sfreq_hz": input_sfreq,
                        "input_n_channels": len(input_raw.ch_names),
                        "input_n_eeg_channels": len(input_eeg_picks),

                        # EOG/channel-type QC
                        "input_n_eog_channels": len(input_eog_picks),
                        "input_eog_channels": input_eog_channels,
                        "input_channel_types": input_channel_types,

                        "input_n_times": input_n_times,
                        "input_duration_seconds": input_n_times / input_sfreq,
                    })


                    # input_sfreq = float(input_raw.info["sfreq"])
                    # input_n_times = int(input_raw.n_times)
                    # input_eeg_picks = mne.pick_types(
                    #     input_raw.info,
                    #     eeg=True,
                    #     meg=False,
                    #     stim=False,
                    #     misc=False,
                    #     exclude=[],
                    # )
                    # qc_base.update({
                    #     "input_sfreq_hz": input_sfreq,
                    #     "input_n_channels": len(input_raw.ch_names),
                    #     "input_n_eeg_channels": len(input_eeg_picks),
                    #     "input_n_times": input_n_times,
                    #     "input_duration_seconds": input_n_times / input_sfreq,
                    # })



                finally:
                    input_raw.close()

                # ------------------------------------------------------
                # Copy config and replace the load_eeg path
                # ------------------------------------------------------
                cfg = deepcopy(base_config)
                load_step_found = False

                for step in cfg["steps"]:
                    if "load_eeg" not in step:
                        continue

                    load_spec = step["load_eeg"]
                    if load_spec is None:
                        step["load_eeg"] = {
                            "params": {"path": str(fpath)},
                            "verbose": False,
                        }
                    elif "params" in load_spec:
                        load_spec.setdefault("params", {})
                        load_spec["params"]["path"] = str(fpath)
                    else:
                        load_spec["path"] = str(fpath)

                    load_step_found = True
                    break

                if not load_step_found:
                    raise KeyError(
                        "base_config does not contain a 'load_eeg' step."
                    )

                # ------------------------------------------------------
                # Run preprocessing once for this physical recording
                # ------------------------------------------------------
                state = eeg_preprocess_pipeline(cfg)

                # Resolve either epochs_clean OR epochs_clean_by_condition.
                epochs_output, condition_mode = _resolve_final_epochs_output(
                    state,
                    final_epochs_key=final_epochs_key,
                )

                if condition_mode:
                    epoch_items = list(epochs_output.items())
                else:
                    epoch_items = [(None, epochs_output)]

                reference_epochs = epoch_items[0][1]

                # All logical conditions from one file must share EEG layout.
                for condition, epochs in epoch_items:
                    if (
                        abs(
                            float(epochs.info["sfreq"])
                            - float(reference_epochs.info["sfreq"])
                        ) > 1e-6
                    ):
                        raise ValueError(
                            f"Condition '{condition}' has a different sampling rate."
                        )
                    if list(epochs.ch_names) != list(reference_epochs.ch_names):
                        raise ValueError(
                            f"Condition '{condition}' has different channel names/order."
                        )

                # # ------------------------------------------------------
                # # Shared preprocessing QC
                # # ------------------------------------------------------
                # bad_channels = [
                #     str(channel)
                #     for channel in state.get("bad_channels", [])
                # ]


                # demoted_unlocalized = [
                #     str(channel)
                #     for channel in state.get("demoted_unlocalized", [])
                # ]

                # ------------------------------------------------------
                # Shared preprocessing QC
                # ------------------------------------------------------

                # MAD detects extreme high-variability channels before RANSAC.
                # The standalone MAD preprocessing step stores its detected
                # channel names separately for QC and traceability.
                mad_bad_channels = [
                    str(channel)
                    for channel in state.get(
                        "mad_bad_channels",
                        [],
                    )
                ]

                # The existing RANSAC preprocessing step stores the channels
                # it detects in state["bad_channels"].
                #
                # Keep a separate copy here so the RANSAC detections remain
                # distinguishable from the MAD detections in downstream QC.
                ransac_bad_channels = [
                    str(channel)
                    for channel in state.get(
                        "bad_channels",
                        [],
                    )
                ]

                # Build the final unique set of channels that were interpolated
                # by either bad-channel detection method.
                #
                # dict.fromkeys() removes duplicates while preserving detection order.
                bad_channels = list(
                    dict.fromkeys(
                        mad_bad_channels
                        + ransac_bad_channels
                    )
                )

                demoted_unlocalized = [
                    str(channel)
                    for channel in state.get(
                        "demoted_unlocalized",
                        [],
                    )
                ]


                excluded_ics = [
                    int(component)
                    for component in state.get("excluded_ics", [])
                ]

                ica = state.get("ica")
                ica_n_components = getattr(ica, "n_components_", None)
                if ica_n_components is not None:
                    ica_n_components = int(ica_n_components)

                excluded_ic_labels: list[str] = []
                iclabel_df = state.get("iclabel_df")
                if (
                    isinstance(iclabel_df, pd.DataFrame)
                    and {"label", "excluded"}.issubset(iclabel_df.columns)
                ):
                    excluded_ic_labels = (
                        iclabel_df.loc[
                            iclabel_df["excluded"].astype(bool),
                            "label",
                        ]
                        .astype(str)
                        .tolist()
                    )

                # final_raw = state.get("raw")
                # final_raw_duration_seconds = (
                #     final_raw.n_times / float(final_raw.info["sfreq"])
                #     if isinstance(final_raw, mne.io.BaseRaw)
                #     else None
                # )



                # ------------------------------------------------------
                # Capture final Raw channel information and EOG QC
                # ------------------------------------------------------
                final_raw = state.get("raw")

                final_eog_channels: list[str] = []
                final_channel_types: dict[str, str] = {}

                if isinstance(final_raw, mne.io.BaseRaw):
                    final_raw_duration_seconds = (
                        final_raw.n_times
                        / float(final_raw.info["sfreq"])
                    )

                    # Automatically identify every channel typed as EOG.
                    final_eog_picks = mne.pick_types(
                        final_raw.info,
                        eeg=False,
                        eog=True,
                        meg=False,
                        stim=False,
                        misc=False,
                        exclude=[],
                    )

                    final_eog_channels = [
                        final_raw.ch_names[idx]
                        for idx in final_eog_picks
                    ]

                    # Preserve the complete final channel-name -> type mapping.
                    final_channel_types = dict(
                        zip(
                            final_raw.ch_names,
                            final_raw.get_channel_types(),
                        )
                    )

                else:
                    final_raw_duration_seconds = None


                # ------------------------------------------------------
                # Capture EOG-supported ICA validation information
                # ------------------------------------------------------
                eog_validation = state.get(
                    "eog_validation_summary",
                    {},
                )

                if not isinstance(eog_validation, Mapping):
                    eog_validation = {}

                eog_channels = [
                    str(channel)
                    for channel in eog_validation.get(
                        "eog_channels",
                        [],
                    )
                ]

                eog_candidate_ics = [
                    int(component)
                    for component in eog_validation.get(
                        "eog_candidate_ics",
                        [],
                    )
                ]

                eog_available = bool(
                    eog_validation.get(
                        "eog_available",
                        False,
                    )
                )

                n_eog_candidate_ics = int(
                    eog_validation.get(
                        "n_eog_candidate_ics",
                        len(eog_candidate_ics),
                    )
                )


                # ------------------------------------------------------
                # Establish/compare reference EEG layout
                # ------------------------------------------------------
                final_sfreq_reference = float(reference_epochs.info["sfreq"])
                final_channels_reference = list(reference_epochs.ch_names)

                if not eeg_info:
                    eeg_info = {
                        "sfreq": final_sfreq_reference,
                        "ch_names": final_channels_reference,
                    }
                    sfreq_matches_reference = True
                    channels_match_reference = True
                else:
                    sfreq_matches_reference = (
                        abs(final_sfreq_reference - eeg_info["sfreq"]) <= 1e-6
                    )
                    channels_match_reference = (
                        final_channels_reference == eeg_info["ch_names"]
                    )

                    if not sfreq_matches_reference:
                        print(
                            f"[warn] sfreq mismatch in '{fpath}': "
                            f"got {final_sfreq_reference}, "
                            f"expected {eeg_info['sfreq']}."
                        )
                    if not channels_match_reference:
                        print(
                            f"[warn] ch_names mismatch in '{fpath}'. "
                            "Using channel names from the first success."
                        )

                qc_notes: list[str] = []
                if not sfreq_matches_reference:
                    qc_notes.append(
                        "Sampling rate differs from the first successful recording."
                    )
                if not channels_match_reference:
                    qc_notes.append(
                        "Channel names or ordering differ from the first successful recording."
                    )
                qc_flag = "pass" if not qc_notes else "review"

                # ------------------------------------------------------
                # Build one QC row per logical analysis condition
                # ------------------------------------------------------
                pending_qc_rows: list[dict[str, Any]] = []

                for condition_idx, (condition, epochs) in enumerate(epoch_items):
                    if condition_mode:
                        condition = str(condition)
                        epoch_qc = _build_condition_epoch_qc(
                            state,
                            condition,
                            epochs,
                            condition_to_eye_state=condition_to_eye_state,
                        )
                        logical_recording_id = f"{recording_id}__{condition}"
                        threshold_key = (
                            "bad_epoch_reject_thresholds_by_condition"
                            if "bad_epoch_reject_thresholds_by_condition" in state
                            else None
                        )
                    else:
                        epoch_qc, threshold_key = _build_single_epoch_qc(
                            state,
                            epochs,
                        )
                        logical_recording_id = recording_id


                    # ------------------------------------------------------
                    # Build final condition-specific recording QC row
                    # ------------------------------------------------------
                    qc_row = dict(qc_base)

                    qc_row.update({
                        # Recording identity
                        "recording_id": logical_recording_id,              # Logical recording ID, e.g. source__EO
                        "source_recording_id": recording_id,               # Original physical recording ID
                        "condition_mode": bool(condition_mode),            # Whether EO/EC condition splitting was used
                        "condition_idx": (
                            condition_idx if condition_mode else None
                        ),                                                 # Index of condition within physical recording
                        "label_idx": per_label_counts[label],              # Recording index within label
                        "global_idx": global_count,                        # Recording index across full batch

                        # Processing status
                        "processing_status": "success",                    # Successful preprocessing indicator
                        "processing_error": None,                          # Error message; None when successful

                        # Final recording/channel information
                        "final_raw_duration_seconds":
                            final_raw_duration_seconds,                    # Final Raw duration after preprocessing
                        "final_n_eog_channels":
                            len(final_eog_channels),                       # Number of channels typed as EOG
                        "final_eog_channels":
                            list(final_eog_channels),                      # Names of final EOG channels
                        "final_channel_types":
                            dict(final_channel_types),                     # Final channel-name -> type mapping

                        # Bad-channel / spatial QC
                        "mad_bad_channels":
                            list(mad_bad_channels),                        # Channels detected by MAD amplitude-outlier detection
                        "n_mad_bad_channels":
                            len(mad_bad_channels),                         # Number of MAD-detected channels

                        "ransac_bad_channels":
                            list(ransac_bad_channels),                     # Channels detected by RANSAC
                        "n_ransac_bad_channels":
                            len(ransac_bad_channels),                      # Number of RANSAC-detected channels

                        "bad_channels":
                            list(bad_channels),                            # Final unique union of MAD + RANSAC interpolated channels
                        "n_bad_channels":
                            len(bad_channels),                             # Total number of unique interpolated channels

                        "demoted_unlocalized":
                            demoted_unlocalized,                           # EEG channels demoted due to missing xyz
                        "n_demoted_unlocalized":
                            len(demoted_unlocalized),                      # Number of demoted unlocalized channels






                        # ICA / ICLabel QC
                        "ica_n_components": ica_n_components,             # Number of fitted ICA components
                        "excluded_ics": excluded_ics,                     # ICA components actually removed
                        "n_excluded_ics": len(excluded_ics),              # Number of removed ICA components
                        "excluded_ic_labels": excluded_ic_labels,         # ICLabel classes of removed components

                        # EOG-supported ocular-artifact QC
                        "eog_available": eog_available,                   # Whether usable EOG channels were available
                        "eog_channels": list(eog_channels),               # EOG channels used for ICA validation
                        "eog_candidate_ics":
                            list(eog_candidate_ics),                       # ICs independently identified by EOG
                        "n_eog_candidate_ics":
                            n_eog_candidate_ics,                           # Number of EOG-supported candidate ICs

                        # Epoch rejection / scaling
                        "rejection_threshold_state_key":
                            threshold_key,                                # State key containing rejection threshold
                        "scale_factor": state.get("scale_factor"),        # EEG amplitude scale factor applied

                        # Processing-method flags
                        "csd_applied":
                            bool(state.get("csd_applied", False)),        # Whether CSD/Laplacian was applied

                        # Cross-recording consistency checks
                        "sfreq_matches_reference":
                            sfreq_matches_reference,                       # Sampling rate matches reference recording
                        "channels_match_reference":
                            channels_match_reference,                      # EEG channel set matches reference

                        # Overall QC assessment
                        "qc_flag": qc_flag,                               # Current recording-level QC status
                        "qc_notes": list(qc_notes),                       # Human-readable QC notes
                    })

                    # Add condition-specific epoch QC:
                    # attempted/rejected/retained epochs, usable minutes, etc.
                    qc_row.update(epoch_qc)

                    # Preserve general condition metadata.
                    if condition_mode:
                        qc_row["analysis_condition"] = str(condition)

                    pending_qc_rows.append(qc_row)

                # ------------------------------------------------------
                # Store outputs only after all condition QC succeeds
                # ------------------------------------------------------
                processing_seconds = perf_counter() - processing_start
                first_qc_idx = len(qc_records)
                qc_indices: list[int] = []

                for offset, qc_row in enumerate(pending_qc_rows):
                    qc_idx = first_qc_idx + offset
                    qc_row["qc_idx"] = qc_idx
                    qc_row["processing_seconds"] = processing_seconds
                    qc_records.append(qc_row)
                    qc_indices.append(qc_idx)

                results[label].append(epochs_output)

                metadata.append({
                    "file_path": str(fpath),
                    "recording_id": recording_id,
                    "source_recording_id": recording_id,
                    "label": label,
                    "subject_id": subject_id,
                    "timepoint": timepoint,
                    "condition_mode": bool(condition_mode),
                    "analysis_conditions": (
                        [str(condition) for condition, _ in epoch_items]
                        if condition_mode
                        else []
                    ),
                    "label_idx": per_label_counts[label],
                    "global_idx": global_count,
                    "qc_idx": qc_indices[0],
                    "qc_indices": qc_indices,
                })

                per_label_counts[label] += 1
                global_count += 1

            except Exception as exc:
                # Failed physical recording -> one failure QC row.
                qc_base["processing_error"] = f"{type(exc).__name__}: {exc}"
                qc_base["qc_notes"] = ["Preprocessing failed."]
                qc_base["processing_seconds"] = perf_counter() - processing_start
                qc_base["qc_idx"] = len(qc_records)
                qc_records.append(qc_base)

                print(
                    f"[warn] Skipping '{fpath}' ({label}): {exc}"
                )

    # ------------------------------------------------------------------
    # Batch summary
    # ------------------------------------------------------------------
    if verbose:
        n_success = sum(
            record["processing_status"] == "success"
            for record in qc_records
        )
        n_failed = sum(
            record["processing_status"] == "failed"
            for record in qc_records
        )
        n_review = sum(
            record["qc_flag"] == "review"
            for record in qc_records
        )

        print("\n" + "=" * 80)
        print("Batch preprocessing summary")
        print("=" * 80)
        print(f"Physical recordings attempted: {n_discovered}")
        print(f"Successful analysis records:   {n_success}")
        print(f"Failed physical recordings:    {n_failed}")
        print(f"Analysis records for review:   {n_review}")
        print("=" * 80)

    return dict(results), metadata, eeg_info, qc_records



def build_epoch_label_lists_multiclass(
    label_to_subjects: Dict[str, List[Any]],
    metadata: List[Dict[str, Any]],
    label_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[List[np.ndarray], List[List[int]], Dict[str, int], List[Tuple[str, str]]]:
    """
    Build per-subject data/labels with stable subject keys, preserving original order.

    Returns (in order of metadata):
      - data_list:  [(n_epochs_i, n_channels, n_times)] per subject
      - label_list: [[lab_id] * n_epochs_i] per subject
      - label_to_id: mapping used
      - subject_keys: [(label, subject_id)] aligned 1:1 with data_list
    """
    if not label_to_subjects or not metadata:
        return [], [], {}, []

    labels_in_data = set(label_to_subjects.keys())

    # Decide mapping (override if provided, otherwise alphabetical default)
    if label_to_id is None:
        labels_sorted = sorted(labels_in_data)
        label_to_id_used = {lab: i for i, lab in enumerate(labels_sorted)}
    else:
        if not isinstance(label_to_id, dict):
            raise TypeError("label_to_id must be a dict like {'ASD': 1, 'TD': 0}")

        # Validate coverage: must include all labels present in label_to_subjects
        missing = labels_in_data - set(label_to_id.keys())
        if missing:
            raise ValueError(
                f"label_to_id is missing labels present in data: {sorted(missing)}"
            )

        # Validate values: ints + unique
        for k, v in label_to_id.items():
            if not isinstance(v, int):
                raise TypeError(f"label_to_id['{k}'] must be an int, got {type(v).__name__}")

        vals = [label_to_id[lab] for lab in labels_in_data]
        if len(set(vals)) != len(vals):
            raise ValueError(
                f"label_to_id values must be unique for labels {sorted(labels_in_data)}; "
                f"got { {lab: label_to_id[lab] for lab in sorted(labels_in_data)} }"
            )

        # (Optional) warn/ignore extra keys not in data
        label_to_id_used = dict(label_to_id)

    data_list: List[np.ndarray] = []
    label_list: List[List[int]] = []
    subject_keys: List[Tuple[str, str]] = []

    # IMPORTANT: do NOT sort metadata; iterate as created
    for m in metadata:
        lab = m["label"]
        sid = m["subject_id"]
        idx = int(m["label_idx"])  # position within this label's list

        if lab not in label_to_subjects:
            raise KeyError(f"Metadata label '{lab}' not found in label_to_subjects keys {sorted(labels_in_data)}")
        if lab not in label_to_id_used:
            raise KeyError(f"Metadata label '{lab}' not found in label_to_id mapping")

        # Fetch the exact subject by index; raises IndexError if misaligned
        subj_item = label_to_subjects[lab][idx]

        # Normalize to ndarray (n_epochs, n_channels, n_times)
        X, _, _ = _normalize_epochs_input(subj_item)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (n_epochs, n_channels, n_times), got {X.shape}")

        n_epochs_i = X.shape[0]
        lab_id = label_to_id_used[lab]

        data_list.append(X)
        label_list.append([lab_id] * n_epochs_i)
        subject_keys.append((lab, sid))

    return data_list, label_list, label_to_id_used, subject_keys




def combine_feature_dfs_per_subject(*dfs_dicts: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Concatenate multiple per-subject feature DataFrames (TD, FD, NLD, MS, etc.)
    for each subject.

    Parameters
    ----------
    dfs_dicts : variable number of dict[str, pd.DataFrame]
        Each dict is keyed by 'LABEL_SUBJECTID' and contains a per-subject DF.

    Returns
    -------
    combined_dfs_dict : dict[str, pd.DataFrame]
        Combined DataFrames keyed by 'LABEL_SUBJECTID'.

    Raises
    ------
    ValueError
        If subject keys or row counts don’t match across dictionaries.
    """
    if len(dfs_dicts) == 0:
        return {}

    # Use the keys of the first dict as reference
    ref_keys = set(dfs_dicts[0].keys())
    combined_dfs_dict: Dict[str, pd.DataFrame] = {}

    # Sanity: all dicts must have the same subject keys
    for d in dfs_dicts[1:]:
        if set(d.keys()) != ref_keys:
            raise ValueError("All feature dictionaries must have the same subject keys.")

    # Now concatenate per subject
    for key in ref_keys:
        dfs_for_key = [d[key] for d in dfs_dicts]

        # Optional but important: row count must match per subject
        n_rows = {df.shape[0] for df in dfs_for_key}
        if len(n_rows) != 1:
            raise ValueError(
                f"Row mismatch for subject '{key}': "
                f"got row counts {n_rows}. Check your epoching / feature extraction."
            )

        combined_dfs_dict[key] = pd.concat(dfs_for_key, axis=1)

    return combined_dfs_dict


def process_combined_dfs(
    combined_dfs_dict: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Harmonize per-subject EEG feature tables to a shared feature set.

    This function is meant for situations like yours, where you have a
    dictionary of DataFrames (one per subject/patient), all with the same
    number of rows (e.g. 89 epochs) but potentially different feature
    columns (e.g. some subjects have extra channels or metrics).

    It does two things:

    1. **Build a consistent feature space (intersection of columns)**

       It finds the set of feature columns that are present in *every*
       subject (the intersection of all column sets). For each subject,
       it returns a new DataFrame that keeps only these shared columns.
       This guarantees that all returned DataFrames have:

         - the same columns
         - all values coming from that subject's own EEG
         - no invented or imputed features

       This is what you would typically pass on to downstream modeling.

    2. **Report what each subject is missing (relative to the union)**

       It also computes the union of all columns that appear in *any*
       subject. For each subject, it then records which of those columns
       are **missing** from that subject's original DataFrame.

       Concretely, for a given subject S:

         info_dict[S] = sorted(list(all_columns_union - columns_of_S))

       This tells you:
         - which features exist somewhere in the dataset
           but were never present for this subject
         - useful for debugging and understanding feature extraction
           differences across subjects (e.g. different montages,
           preprocessing versions, etc.)

    Parameters
    ----------
    combined_dfs_dict : dict[str, pd.DataFrame]
        Mapping from subject ID (e.g. 'TD_NDARLA559EGK') to that subject's
        EEG feature DataFrame.

    Returns
    -------
    processed_dfs_dict : dict[str, pd.DataFrame]
        Dictionary with the same keys (subject IDs), where each DataFrame
        contains only the columns that are present in *all* subjects
        (intersection of column sets), with identical column order.

    info_dict : dict[str, list[str]]
        For each subject ID, a list of feature names that exist in at least
        one other subject but are missing from this subject's original
        DataFrame (i.e., this subject's "gaps" relative to the union of
        all columns).
    """

    if not combined_dfs_dict:
        return {}, {}

    # 1. Collect column sets per subject
    col_sets: Dict[str, set] = {
        subj: set(df.columns) for subj, df in combined_dfs_dict.items()
    }

    # 2. Intersection (shared columns across all subjects)
    common_cols = set.intersection(*col_sets.values())
    common_cols_sorted: List[str] = sorted(common_cols)

    # 3. Union (for "what is this subject missing vs others?")
    all_cols = set().union(*col_sets.values())

    # 4. Build processed dict using only intersection columns
    processed_dfs_dict: Dict[str, pd.DataFrame] = {
        subj: df[common_cols_sorted].copy()
        for subj, df in combined_dfs_dict.items()
    }

    # 5. Build info dict: only "what is this subject missing vs union?"
    info_dict: Dict[str, List[str]] = {}
    for subj, cols in col_sets.items():
        missing_relative_to_union = sorted(all_cols - cols)
        info_dict[subj] = missing_relative_to_union

    return processed_dfs_dict, info_dict

    
def stack_features_with_groups(
    combined_dfs_dict: dict[str, pd.DataFrame],
    label_list: list[list[int]],
    subject_keys: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[str, str]], list[str]]:
    """
    Stack per-subject combined feature DataFrames into a single NumPy matrix.

    Returns
    -------
    X_raw : np.ndarray
        Stacked feature matrix (sum(n_epochs_i), n_features)

    y : np.ndarray
        Stacked label vector (sum(n_epochs_i),)

    groups : np.ndarray
        Group ID per epoch (sum(n_epochs_i),)

    group_id_to_key : dict[int, (label, subject_id)]
        Mapping from group integer → subject identity

    feature_names : list[str]
        Column names corresponding to the feature dimensions in X_raw
    """

    # --- Basic alignment check ---
    if len(subject_keys) != len(label_list):
        raise ValueError(
            f"subject_keys and label_list must be same length. "
            f"Got {len(subject_keys)} vs {len(label_list)}."
        )

    # --- Extract consistent feature names from any subject ---
    if not combined_dfs_dict:
        raise ValueError("combined_dfs_dict is empty — cannot extract features.")

    example_key = next(iter(combined_dfs_dict))
    feature_names = combined_dfs_dict[example_key].columns.tolist()

    # Containers
    X_list, y_list, g_list = [], [], []
    group_id_to_key = {}

    # --- Loop through subjects in the correct original order ---
    for i, ((label, subj_id), y_i) in enumerate(zip(subject_keys, label_list)):
        dict_key = f"{label}_{subj_id}"

        if dict_key not in combined_dfs_dict:
            raise ValueError(f"Missing combined features for subject '{dict_key}'")

        df = combined_dfs_dict[dict_key]

        # Ensure column order matches
        if df.columns.tolist() != feature_names:
            raise ValueError(f"Column mismatch for subject {dict_key}. "
                             "Feature sets differ across subjects.")

        X_i = df.values
        n_epochs_i = X_i.shape[0]

        if len(y_i) != n_epochs_i:
            raise ValueError(
                f"Label mismatch for subject {dict_key}: "
                f"{len(y_i)} labels vs {n_epochs_i} epochs."
            )

        # Append
        X_list.append(X_i)
        y_list.append(np.asarray(y_i, dtype=int))
        g_list.append(np.full(n_epochs_i, i, dtype=int))
        group_id_to_key[i] = (label, subj_id)

    # Final stacked outputs
    X_raw = np.vstack(X_list)
    y = np.hstack(y_list)
    groups = np.hstack(g_list)

    return X_raw, y, groups, group_id_to_key, feature_names



# ---------------------------
# Functions to save and load 
# ---------------------------
def save_feature_extraction_outputs(
    output_dir: Union[str, Path],
    dfs_dict: Dict[str, pd.DataFrame],
    label_list: List[List[int]],
    subject_keys: List[Tuple[str, str]],
    *,
    cols: Optional[List[str]] = None,
    prefix: str = "features",
    compress: bool = True,
) -> Path:
    """
    Generic saver for feature-extraction outputs.

    Saves:
      - {prefix}_dfs_dict.pkl(.gz) : Dict[str, DataFrame]
      - {prefix}_label_list.npy    : ragged List[List[int]] as numpy object array
      - {prefix}_subject_keys.json : List[[label, subject_id], ...]
      - {prefix}_cols.json         : optional feature column names

    Parameters
    ----------
    output_dir : str | Path
        Directory to save into.
    dfs_dict : dict[str, pd.DataFrame]
        Per-subject dataframes keyed by e.g. "LABEL_SUBJECTID".
    label_list : list[list[int]]
        Ragged list: labels per epoch for each subject.
    subject_keys : list[tuple[str, str]]
        [(label, subject_id), ...] aligned with label_list order.
    cols : list[str], optional
        Feature columns (ordered), if you want to persist schema.
    prefix : str
        Allows multiple feature families in one folder (e.g., "nd", "psd", "conn").
    compress : bool
        If True, gzip the pickle for dfs_dict.

    Returns
    -------
    Path
        The output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Save dfs_dict (pickle preserves DataFrame dtypes/index)
    pkl_path = out / (f"{prefix}_dfs_dict.pkl.gz" if compress else f"{prefix}_dfs_dict.pkl")
    if compress:
        import gzip
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(dfs_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(dfs_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2) Save label_list (ragged) as numpy object array
    np.save(out / f"{prefix}_label_list.npy", np.array(label_list, dtype=object), allow_pickle=True)

    # 3) Save subject_keys as JSON (tuples -> lists)
    subject_keys_json = [[lab, sid] for (lab, sid) in subject_keys]
    with open(out / f"{prefix}_subject_keys.json", "w") as f:
        json.dump(subject_keys_json, f, indent=2)

    # 4) Optional cols schema
    if cols is not None:
        with open(out / f"{prefix}_cols.json", "w") as f:
            json.dump(list(cols), f, indent=2)

    print(f"✅ Saved extraction outputs to: {out.resolve()}")
    return out


def load_feature_extraction_outputs(
    output_dir: Union[str, Path],
    *,
    prefix: str = "features",
    compress: bool = True,
    expect_cols: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], List[List[int]], List[Tuple[str, str]], Optional[List[str]]]:
    """
    Generic loader matching save_feature_extraction_outputs().
    """
    out = Path(output_dir)

    # 1) Load dfs_dict
    pkl_path = out / (f"{prefix}_dfs_dict.pkl.gz" if compress else f"{prefix}_dfs_dict.pkl")
    if compress:
        import gzip
        with gzip.open(pkl_path, "rb") as f:
            dfs_dict = pickle.load(f)
    else:
        with open(pkl_path, "rb") as f:
            dfs_dict = pickle.load(f)

    # 2) Load label_list
    label_arr = np.load(out / f"{prefix}_label_list.npy", allow_pickle=True)
    label_list = label_arr.tolist()

    # 3) Load subject_keys
    with open(out / f"{prefix}_subject_keys.json", "r") as f:
        subject_keys_json = json.load(f)
    subject_keys = [(lab, sid) for lab, sid in subject_keys_json]

    # 4) Optional cols
    cols = None
    cols_path = out / f"{prefix}_cols.json"
    if cols_path.exists():
        with open(cols_path, "r") as f:
            cols = json.load(f)
    elif expect_cols:
        raise FileNotFoundError(f"Expected {cols_path.name} but it was not found in {out}")

    return dfs_dict, label_list, subject_keys, cols
   

def save_prepared_dataset_bundle(
    output_dir: Union[str, Path],
    *,
    X_raw: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    label_to_id: Optional[Dict[str, int]] = None,
    group_id_to_key: Optional[Dict[int, Tuple[str, str]]] = None,
    feature_names: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    info_dict: Optional[Dict[str, List[str]]] = None,
    prefix: str = "prepared",
    cast: bool = True,
) -> Path:
    """
    Save a "prepared dataset" bundle. Writes only the artifacts you provide.

    Files (only if corresponding input is not None)
    ------------------------------------------------
    - {prefix}_X_raw.npy
    - {prefix}_y.npy
    - {prefix}_groups.npy
    - {prefix}_label_to_id.json
    - {prefix}_group_id_to_key.json
    - {prefix}_feature_names.json
    - {prefix}_metadata.csv
    - {prefix}_missing_features_by_subject.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = []

    # Arrays
    if X_raw is not None:
        arr = X_raw.astype(np.float32) if cast else X_raw
        np.save(out / f"{prefix}_X_raw.npy", arr)
        saved.append(f"{prefix}_X_raw.npy")

    if y is not None:
        arr = y.astype(np.int32) if cast else y
        np.save(out / f"{prefix}_y.npy", arr)
        saved.append(f"{prefix}_y.npy")

    if groups is not None:
        arr = groups.astype(np.int32) if cast else groups
        np.save(out / f"{prefix}_groups.npy", arr)
        saved.append(f"{prefix}_groups.npy")

    # JSON mappings
    if label_to_id is not None:
        with open(out / f"{prefix}_label_to_id.json", "w") as f:
            json.dump(label_to_id, f, indent=2)
        saved.append(f"{prefix}_label_to_id.json")

    if group_id_to_key is not None:
        # int keys -> str keys; tuple -> dict for clean JSON
        group_serializable = {
            str(gid): {"label": lab, "subject_id": sid}
            for gid, (lab, sid) in group_id_to_key.items()
        }
        with open(out / f"{prefix}_group_id_to_key.json", "w") as f:
            json.dump(group_serializable, f, indent=2)
        saved.append(f"{prefix}_group_id_to_key.json")

    if feature_names is not None:
        with open(out / f"{prefix}_feature_names.json", "w") as f:
            json.dump(list(feature_names), f, indent=2)
        saved.append(f"{prefix}_feature_names.json")

    # Metadata CSV
    if metadata is not None:
        pd.DataFrame(metadata).to_csv(out / f"{prefix}_metadata.csv", index=False)
        saved.append(f"{prefix}_metadata.csv")

    # Optional info dict
    if info_dict is not None:
        with open(out / f"{prefix}_missing_features_by_subject.json", "w") as f:
            json.dump(info_dict, f, indent=2)
        saved.append(f"{prefix}_missing_features_by_subject.json")

    if saved:
        print(f"✅ Saved {len(saved)} artifact(s) to: {out.resolve()}")
        # (optional) print which ones
        # print("   " + "\n   ".join(saved))
    else:
        print(f"⚠️ Nothing was saved (all inputs were None). Directory: {out.resolve()}")

    return out


def load_prepared_dataset_bundle(
    output_dir: Union[str, Path],
    *,
    prefix: str = "prepared",
    require: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load a prepared dataset bundle. Loads whatever files exist.

    Returns a dict with keys:
      X_raw, y, groups, label_to_id, group_id_to_key,
      feature_names, metadata, info_dict

    If `require` is provided, it should be a list of keys that must be present,
    e.g. require=["X_raw","y","groups"].
    """
    out = Path(output_dir)

    bundle: Dict[str, Any] = {
        "X_raw": None,
        "y": None,
        "groups": None,
        "label_to_id": None,
        "group_id_to_key": None,
        "feature_names": None,
        "metadata": None,
        "info_dict": None,
    }

    # Arrays
    xf_path = out / f"{prefix}_X_raw.npy"
    if xf_path.exists():
        bundle["X_raw"] = np.load(xf_path, allow_pickle=False)

    y_path = out / f"{prefix}_y.npy"
    if y_path.exists():
        bundle["y"] = np.load(y_path, allow_pickle=False)

    g_path = out / f"{prefix}_groups.npy"
    if g_path.exists():
        bundle["groups"] = np.load(g_path, allow_pickle=False)

    # JSON
    l2i_path = out / f"{prefix}_label_to_id.json"
    if l2i_path.exists():
        with open(l2i_path, "r") as f:
            bundle["label_to_id"] = json.load(f)

    g2k_path = out / f"{prefix}_group_id_to_key.json"
    if g2k_path.exists():
        with open(g2k_path, "r") as f:
            raw = json.load(f)
        # convert back to Dict[int, Tuple[str,str]]
        bundle["group_id_to_key"] = {
            int(gid): (v["label"], v["subject_id"]) for gid, v in raw.items()
        }

    fn_path = out / f"{prefix}_feature_names.json"
    if fn_path.exists():
        with open(fn_path, "r") as f:
            bundle["feature_names"] = json.load(f)

    info_path = out / f"{prefix}_missing_features_by_subject.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            bundle["info_dict"] = json.load(f)

    # Metadata CSV
    meta_path = out / f"{prefix}_metadata.csv"
    if meta_path.exists():
        bundle["metadata"] = pd.read_csv(meta_path).to_dict(orient="records")

    # Require checks
    if require:
        missing = [k for k in require if bundle.get(k) is None]
        if missing:
            raise FileNotFoundError(
                f"Missing required artifacts in {out.resolve()}: {missing}"
            )

    return bundle


def save_prepared_dataset(
    output_dir: str | Path,
    X_features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_to_id: Dict[str, int],
    group_id_to_key: Dict[int, Tuple[str, str]],
    feature_names: List[str],
    metadata: List[Dict[str, Any]],
    info_dict: Optional[Dict[str, List[str]]] = None,
):
    """
    Save processed FEATURE-LEVEL EEG dataset into a directory.

    Files created
    -------------
    - X_features.npy                 : (n_epochs, n_features) feature matrix
    - y_labels.npy                   : (n_epochs,) integer class labels
    - groups.npy                     : (n_epochs,) group IDs (subject index)
    - label_to_id.json               : mapping from label string -> int
    - group_id_to_key.json           : mapping from group ID -> {label, subject_id}
    - feature_names.json             : ordered list of feature column names
    - metadata.csv                   : original metadata (one row per subject/file)
    - missing_features_by_subject.json (optional):
        if `info_dict` is provided, this JSON maps
        subject_id -> list of feature names that were present in at least
        one other subject but missing for this subject (i.e., gaps relative
        to the union of all columns before taking the intersection).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save main arrays ---
    np.save(output_dir / "X_features.npy", X_features.astype(np.float32))
    np.save(output_dir / "y_labels.npy",  y.astype(np.int32))
    np.save(output_dir / "groups.npy",    groups.astype(np.int32))

    # --- Save label mapping ---
    with open(output_dir / "label_to_id.json", "w") as f:
        json.dump(label_to_id, f, indent=2)

    # --- Save group_id_to_key (convert keys + tuples to nice JSON) ---
    group_serializable = {
        str(gid): {"label": lab, "subject_id": sid}
        for gid, (lab, sid) in group_id_to_key.items()
    }
    with open(output_dir / "group_id_to_key.json", "w") as f:
        json.dump(group_serializable, f, indent=2)

    # --- Save feature names (ordered to match X_features columns) ---
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # --- Save metadata as CSV ---
    pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)

    # --- Save missing-features info, if provided ---
    if info_dict is not None:
        with open(output_dir / "missing_features_by_subject.json", "w") as f:
            json.dump(info_dict, f, indent=2)

    print(f"✅ Feature dataset saved successfully to: {output_dir.resolve()}")


def load_prepared_dataset(
    input_dir: str | Path,
) -> tuple[
    np.ndarray,                    # X_features
    np.ndarray,                    # y
    np.ndarray,                    # groups
    Dict[str, int],                # label_to_id
    Dict[int, Tuple[str, str]],    # group_id_to_key
    List[str],                     # feature_names
    List[Dict[str, Any]],          # metadata
    Optional[Dict[str, List[str]]],# missing_features_by_subject (info_dict)
]:
    """
    Load a feature-level EEG dataset previously saved with `save_prepared_dataset`.

    Expects the following files inside `input_dir`:
        - X_features.npy
        - y_labels.npy
        - groups.npy
        - label_to_id.json
        - group_id_to_key.json
        - feature_names.json
        - metadata.csv
        - missing_features_by_subject.json (optional)

    Returns
    -------
    X_features : np.ndarray, shape (n_epochs, n_features)
        Stacked feature matrix.
    y : np.ndarray, shape (n_epochs,)
        Integer class labels.
    groups : np.ndarray, shape (n_epochs,)
        Group IDs (subject index per epoch).
    label_to_id : dict[str, int]
        Mapping from label string to integer ID.
    group_id_to_key : dict[int, (str, str)]
        Mapping from group ID -> (label, subject_id).
    feature_names : list[str]
        Ordered feature names corresponding to columns of X_features.
    metadata : list[dict]
        Original metadata rows (one per subject/file).
    missing_features_by_subject : dict[str, list[str]] or None
        If present, maps each subject_id to the list of feature names that
        were present in at least one other subject but missing for this
        subject before taking the intersection (i.e., the saved `info_dict`).
        If the JSON file is not found, this will be None.
    """
    input_dir = Path(input_dir)

    # --- Load main arrays ---
    X_features = np.load(input_dir / "X_features.npy")
    y = np.load(input_dir / "y_labels.npy")
    groups = np.load(input_dir / "groups.npy")

    # --- Load label mapping ---
    with open(input_dir / "label_to_id.json", "r") as f:
        label_to_id: Dict[str, int] = json.load(f)

    # --- Load group_id_to_key and convert JSON keys back to int ---
    with open(input_dir / "group_id_to_key.json", "r") as f:
        raw_group_map = json.load(f)

    group_id_to_key: Dict[int, Tuple[str, str]] = {
        int(gid): (entry["label"], entry["subject_id"])
        for gid, entry in raw_group_map.items()
    }

    # --- Load feature names ---
    with open(input_dir / "feature_names.json", "r") as f:
        feature_names: List[str] = json.load(f)

    # --- Load metadata ---
    metadata_df = pd.read_csv(input_dir / "metadata.csv")
    metadata: List[Dict[str, Any]] = metadata_df.to_dict(orient="records")

    # --- Load missing-features info_dict, if present ---
    missing_features_path = input_dir / "missing_features_by_subject.json"
    if missing_features_path.exists():
        with open(missing_features_path, "r") as f:
            missing_features_by_subject: Dict[str, List[str]] = json.load(f)
    else:
        missing_features_by_subject = None

    return (
        X_features,
        y,
        groups,
        label_to_id,
        group_id_to_key,
        feature_names,
        metadata,
        missing_features_by_subject,
    )




# ---------------------------
# Light DataFrame helpers
# ---------------------------
def count_feature_columns(df: pd.DataFrame, features=("mean", "std", "rms", "kurt")) -> pd.DataFrame:
    """
    Count how many columns in the feature DataFrame correspond to each feature type.

    This scans column names (e.g., 'T2_mean__Fp1', 'T7_hj_mob__Fz') and counts how many
    contain each keyword in `features`. Returns a one-row DataFrame summarizing counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing extracted EEG features (columns like 'T2_mean__Fp1').
    features : tuple or list of str
        Feature keywords to search for within column names (e.g., ("mean", "std", "rms", "kurt")).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with one column per feature keyword and values equal to
        the number of matching columns in `df`.

    Example
    -------
    >>> df.columns = ['T2_mean__Fp1', 'T2_mean__Fp2', 'T3_std__Fp1']
    >>> count_feature_columns(df, features=('mean','std','rms'))
       mean  std
    0     2    1
    """
    counts = {}
    for feat in features:
        matches = [c for c in df.columns if feat in c]
        if matches:
            counts[feat] = len(matches)

    return pd.DataFrame([counts])



def feature_help_table(registry: dict, docs: dict | None = None, domain: str | None = None) -> pd.DataFrame:
    """
    Build a simple table of allowed feature codes for a domain.

    Columns:
      - code: the key you pass in "features=()"
      - function: the Python function that runs
      - description: one-liner (from your docs dict; optional)

    Usage:
      feature_help_table(FEATURE_REGISTRY_TIME, FEATURE_DOCS_TIME, domain="Time")
    """
    rows = []
    for code, fn in registry.items():
        rows.append({
            "code": code,
            "function": getattr(fn, "__name__", str(fn)),
            "description": (docs or {}).get(code, "")
        })
    df = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    if domain:
        df.attrs["domain"] = domain
    return df

def validate_feature_names(requested, *, registry, domain: str):
    """
    Check that all requested feature codes exist in the registry.
    Case-insensitive; trims whitespace.
    Raises ValueError listing allowed codes if not.
    """
    if not requested:
        return []
    
    # Normalize to lowercase for comparison
    reg_keys_lower = {k.lower(): k for k in registry.keys()}
    valid = []
    invalid = []

    for r in requested:
        key = reg_keys_lower.get(str(r).strip().lower())
        if key:
            valid.append(key)
        else:
            invalid.append(r)

    if invalid:
        allowed = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"[{domain}] Invalid feature name(s): {invalid}. "
            f"Allowed values: {allowed}"
        )
    
    return valid

