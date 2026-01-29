# utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import h5py
import mne
import json
from collections import defaultdict
from copy import deepcopy
import re
import pickle

from .eeg_preprocess import eeg_preprocess_pipeline, plot_pipeline_text, config_nehemiah
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





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
        .bdf, .edf, .cnt, .set, .fif, .mff, .egi, .raw, .gdf

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
    }

    # Check support
    if ext not in loaders:
        raise ValueError(f"Unsupported EEG extension: '{ext}'")

    # Select correct reader and load file
    loader = loaders[ext]
    raw = loader(p, preload=preload)

    return raw


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
    id_col: str = "UUID",
    label_col: str = "label",
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

    Parameters
    ----------
    in_dir : str or Path
        Root directory to start searching from.

    pattern : str
        File pattern to match, e.g. "*_r.mat", "*_eeg.set".

    metadata_path : str or Path
        Path to the CSV/XLSX metadata file containing subject IDs and labels.
        REQUIRED. If None, an error is raised.

    id_col : str
        Column name in the metadata file for subject IDs.

    label_col : str
        Column name in the metadata file for the label / phenotype.

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

    # ----------------------------------------------------------------------
    # 0) Metadata is required
    # ----------------------------------------------------------------------
    if metadata_path is None:
        raise ValueError("metadata_path is required but was not provided.")

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

    # Nothing found -> return empty DataFrame early
    if not files:
        print(f"No files found in {in_dir} with pattern='{pattern}' and backend='{backend}'.")
        return pd.DataFrame()

    # ----------------------------------------------------------------------
    # 2) Load metadata mapping (subject_id + label)
    # ----------------------------------------------------------------------
    subj_map = _load_subject_mapping(metadata_path, id_col=id_col, label_col=label_col)
    if subj_map is None:
        raise ValueError(f"Could not load metadata file: {metadata_path}")

    # Convert to pandas "string" dtype (keeps missing values as <NA>, not "nan")
    subj_map["subject_id"] = subj_map["subject_id"].astype("string")

    # Optional: normalize numeric-like IDs (only if you turn it on)
    if normalize_numeric_subject_ids:
        subj_map["subject_id"] = subj_map["subject_id"].map(_normalize_subject_id).astype("string")


    # ----------------------------------------------------------------------
    # 3) Process each EEG file
    # ----------------------------------------------------------------------
    for p in tqdm(files, desc="Scanning EEG files", unit="file"):
    
        # Get ID
        file_id_raw  = subject_id_from_borders(p, subject_id_borders)  # e.g. "sub-0026"
        # Build the minimal per-file record (no hardcoded None scaffolding)
        info = {
            "filepath": str(p),
            "filename": p.name,
            "subject_id_raw": file_id_raw,  # Keep the raw ID for readability/debugging
            "subject_id": file_id_raw,   # Use this as the merge key (we will normalize it later if flag is on)
            "error": None,
        }

        # Optional: normalize file-derived IDs too (must match metadata side)
        if normalize_numeric_subject_ids:
            info["subject_id"] = _normalize_subject_id(info["subject_id"])

        try:
            # --------------------------
            # H5 backend: dynamic key loop
            # --------------------------
            if backend == "h5":                
                # Always safe/lightweight: list top-level keys
                info.update(load_h5_keys(p))  # {"keys": [...]}

                # Loop through EVERY key and:
                #  1) try to infer EEG dims from dataset shapes (2D or 3D)
                #  2) read small dataset values into "<key>_value"
                for k in info.get("keys", []):
                    # Lightweight inspection (type, shape, dtype) without loading arrays
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

                        # If dims inference succeeded, store it ONCE (first match wins)
                        # This avoids overwriting dims if multiple datasets exist.
                        if dims and not any(key in info for key in ("n_times", "n_channels", "n_segments")):
                            info.update(dims)

                    # ---- 2) Read small dataset values (scalar-ish) ----
                    # Example: samplingRate stored as 1x1 -> returns 1000.0 after unwrap
                    val = read_h5_dataset_value(p, k, max_elems=max_value_elems)
                    if val is not None:
                        info[f"{k}_value"] = val

                # Compute duration after we've collected dims + samplingRate_value
                info.update(compute_durations(info))
            # --------------------------
            # MNE backend: keep your current approach
            # --------------------------
            elif backend == "mne":
                # If your load_raw_eeg returns a dict, info.update(meta) will just work.
                # If it returns an MNE Raw object, adapt as needed.
                meta = load_raw_eeg(p, preload=False)

                # If you currently return an MNE Raw object, you can compute inline:
                # raw = meta
                # info.update({...})
                #
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

        # Save this file's record
        rows.append(info)

    # Convert to DataFrame (pandas will handle missing keys as NaN)
    df = pd.DataFrame(rows)

    # ----------------------------------------------------------------------
    # 4) Merge label metadata
    #
    # We support two ID-matching modes:
    #   - normalize_numeric_subject_ids=True:
    #       Normalize numeric-like IDs on BOTH sides (e.g., "0046", "sub-0046", 46 -> "46"),
    #       then do an exact merge.
    #   - normalize_numeric_subject_ids=False:
    #       Filenames may contain "decorated" IDs (prefix/suffix/date/run). Resolve each
    #       df ID to a metadata ID using boundary-aware containment (not naive substring),
    #       then do an exact merge.
    # ----------------------------------------------------------------------
    if not df.empty:

        # Ensure consistent dtype for safe merging (<NA> preserved)
        df["subject_id"] = df["subject_id"].astype("string")
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

        # Exact merge (validated)
        df = df.merge(
            subj_map.drop(columns=["subject_id"]),  # prevent duplicate cols
            on="subject_id_merge",
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        # Convert subject_id_merge missing values to np.nan (instead of <NA>)
        df["subject_id_merge"] = (
            df["subject_id_merge"]
            .astype(object)
            .where(df["subject_id_merge"].notna(), np.nan)
        )


        # Human-friendly match status (keep _merge too if you want)
        df["match_status"] = df["_merge"].map({
            "both": "matched",
            "left_only": "unmatched_file_id",
            "right_only": "metadata_only",
        }).astype("string")

        # Clearer summary printout
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
                            ["subject_id_raw", "subject_id", "subject_id_merge", "filename"]
                        ]


    return df, uunmatched_df

def convert_scan_to_mne_fif(
    df: pd.DataFrame,
    out_dir: str | Path = "sample_butler_prepared",
    montage_name: str = "biosemi64",
    overwrite: bool = False,
    preload: bool = True,
    *,
    metadata_path: str | Path,
    id_col: str = "id",
    label_col: str = "label",
    strict: bool = True,
    normalize_numeric_subject_ids: bool = False,  
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert  sample EEGLAB `.set` files (listed in `df`) into MNE FIF files,
    writing outputs into subfolders by metadata label, and returning both the
    conversion plan and the set of unmatched files.

    This function uses the metadata file as the source of truth for:
      1) which subjects are eligible for conversion, and
      2) what label each subject should receive.

    ID matching modes
    -----------------
    normalize_numeric_subject_ids = False (default)
        Treat metadata IDs as canonical string tokens. File-derived IDs may be
        "decorated" (prefix/suffix/date/run markers). We resolve each file ID to
        a metadata ID using `resolve_to_metadata_id()` (boundary-aware containment),
        then do an exact join.

    normalize_numeric_subject_ids = True
        Intended for numeric-like IDs (e.g., 46, "0046", "sub-0046"). We normalize
        BOTH metadata and file IDs via `_normalize_subject_id()` and join exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Scan table describing input EEG files. Must include at least:
        - 'filepath'   : path to the source `.set` file
        - 'filename'   : basename (used for output naming)
        - 'subject_id' : ID extracted during the scan step

    out_dir : str | Path
        Root output directory. Output FIF files are written under:
            out_dir/<label>/<filename_stem>.fif

    montage_name : str
        Name of an MNE standard montage to apply (default "biosemi64"). Channels
        not present in the montage are re-typed as 'misc' so they are retained.

    overwrite : bool
        If True, overwrite existing `.fif` files. If False, existing outputs
        are skipped.

    preload : bool
        Passed to the EEG loader when reading `.set` files.

    metadata_path : str | Path
        Path to a CSV/XLSX metadata file containing canonical subject IDs and labels.

    id_col : str
        Column name in the metadata file containing subject IDs.

    label_col : str
        Column name in the metadata file containing labels.

    strict : bool
        If True, raise an error if filtering by metadata results in zero rows to
        convert. If False, the function returns empty outputs after printing a summary.

    normalize_numeric_subject_ids : bool
        Controls ID matching mode (see "ID matching modes" above).

    Returns
    -------
    converted_df : pd.DataFrame
        Subset of input `df` that matched metadata, with columns:
        - subject_id_key : canonical key used for metadata join
        - label          : label from metadata (source of truth)

    uunmatched_df : pd.DataFrame
        Rows from `df` that could not be matched to metadata (debugging subset).

    Notes
    -----
    This function performs conversion as a side effect (writes FIF files), and
    returns DataFrames to help users inspect what was converted vs. dropped.
    """

    # ------------------------------------------------------------------
    # 0) Validate required df columns
    # ------------------------------------------------------------------
    required_cols = {"filepath", "subject_id", "filename"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[convert_butler_from_scan] df is missing required columns: {sorted(missing_cols)}"
        )

    if metadata_path is None:
        raise ValueError("[convert_butler_from_scan] metadata_path is required.")

    # ------------------------------------------------------------------
    # 1) Load metadata and build (subject_id_key -> label) mapping
    # ------------------------------------------------------------------
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"[convert_butler_from_scan] metadata_path not found: {metadata_path}"
        )

    # Excel vs CSV
    if metadata_path.suffix.lower() in {".xlsx", ".xls"}:
        meta = pd.read_excel(metadata_path)
    else:
        meta = pd.read_csv(metadata_path)

    if id_col not in meta.columns or label_col not in meta.columns:
        raise ValueError(
            f"[convert_butler_from_scan] metadata file must contain columns "
            f"'{id_col}' and '{label_col}'. Found: {list(meta.columns)}"
        )

    meta = meta[[id_col, label_col]].copy()

    # Build canonical metadata join key
    if normalize_numeric_subject_ids:
        meta["subject_id_key"] = meta[id_col].map(_normalize_subject_id)
    else:
        meta["subject_id_key"] = meta[id_col].astype(str).str.strip()

    meta["label_meta"] = meta[label_col]

    # Drop unusable IDs / labels
    meta = meta.dropna(subset=["subject_id_key"])
    meta["label_meta"] = meta["label_meta"].apply(lambda x: np.nan if pd.isna(x) else x)
    meta = meta.dropna(subset=["label_meta"])
    meta = meta[meta["label_meta"].astype(str).str.strip() != ""]

    # Ensure metadata key is unique (required for validate="many_to_one")
    meta = meta.drop_duplicates(subset=["subject_id_key"], keep="first")

    # Optional: warn about collisions in numeric normalization
    if normalize_numeric_subject_ids:
        n_raw_per_key = meta.groupby("subject_id_key")[id_col].nunique()
        if (n_raw_per_key > 1).any():
            example = n_raw_per_key[n_raw_per_key > 1].sort_values(ascending=False).head(10)
            print("[convert_butler_from_scan] Warning: numeric normalization caused ID collisions.")
            print(example)

    # ------------------------------------------------------------------
    # 2) Build df join key and filter to subjects present in metadata
    # ------------------------------------------------------------------
    work = df.copy()

    if normalize_numeric_subject_ids:
        work["subject_id_key"] = work["subject_id"].map(_normalize_subject_id)
    else:
        # Resolve decorated file IDs to canonical metadata IDs (boundary-aware)
        meta_ids = meta["subject_id_key"].dropna().astype(str).unique().tolist()
        work["subject_id_key"] = (
            work["subject_id"].astype(str).str.strip()
            .map(lambda s: resolve_to_metadata_id(s, meta_ids))
        )

    # Unmatched subset (for user debugging)
    uunmatched_df = work.loc[
        work["subject_id_key"].isna() | ~work["subject_id_key"].isin(meta["subject_id_key"]),
        ["subject_id", "subject_id_key", "filename", "filepath"]
    ].copy()

    # Inner join => only keep rows present in metadata
    converted_df = work.merge(
        meta[["subject_id_key", "label_meta"]],
        on="subject_id_key",
        how="inner",
        validate="many_to_one",
    )

    # Metadata is source of truth for label
    converted_df["label"] = converted_df["label_meta"]
    converted_df = converted_df.drop(columns=["label_meta"])

    n_in = len(df)
    n_keep = len(converted_df)
    n_drop = n_in - n_keep
    print(f"[convert_butler_from_scan] Keeping {n_keep}/{n_in} files after metadata filter. Dropped {n_drop}.")

    if strict and n_keep == 0:
        raise ValueError("[convert_butler_from_scan] After metadata filtering, no rows remain to convert.")

    # ------------------------------------------------------------------
    # 3) Prepare output directory and montage
    # ------------------------------------------------------------------
    out_dir_path = Path(out_dir).expanduser().resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        montage = mne.channels.make_standard_montage(montage_name)
    except Exception as exc:
        raise ValueError(
            f"[convert_butler_from_scan] Invalid montage '{montage_name}'."
        ) from exc

    # ------------------------------------------------------------------
    # 4) Conversion loop (write FIFs)
    # ------------------------------------------------------------------
    failed: list[tuple[str, str]] = []

    for row in tqdm(converted_df.itertuples(index=False), total=len(converted_df), desc="Converting EEG → FIF (by label)", unit="file"):
        file_path = Path(row.filepath)
        filename = str(row.filename)
        label = str(row.label).strip()

        label_dir = out_dir_path / label
        label_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem
        out_path = label_dir / f"{stem}.fif"

        if out_path.exists() and not overwrite:
            continue

        try:
            raw = load_raw_eeg(file_path, preload=preload)

            # Keep channels not in montage (type them as misc)
            montage_chs = set(montage.ch_names)
            unknown = [ch for ch in raw.ch_names if ch not in montage_chs]
            if unknown:
                raw.set_channel_types({ch: "misc" for ch in unknown})

            raw.set_montage(montage)
            raw.save(out_path.as_posix(), overwrite=overwrite, verbose=False)

        except Exception as e:
            failed.append((file_path.name, str(e)))

    # ------------------------------------------------------------------
    # 5) Summary
    # ------------------------------------------------------------------
    if failed:
        print("\n[convert_butler_from_scan] Failed to process the following files:")
        for fname, err in failed:
            print(f" - {fname}: {err}")

    return converted_df, uunmatched_df




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
        desc="Converting ABC-CT → FIF (by label)", unit="file"
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


# Old version
# def convert_abcct_from_scan(
#     df: pd.DataFrame,
#     out_dir: str | Path = "abcct_data_prepared",
#     montage_name: str | None = None,
#     overwrite: bool = False,
#     n_blocks_to_keep: int = 3,
#     require_n_blocks: bool = True,
# ) -> None:
#     """
#     Convert ABC-CT *_r.mat resting EEG files (previously discovered via
#     `scan_eeg_directory`) into MNE FIF format, organizing output by label.

#     This function assumes that `df` already contains one row per MATLAB file,
#     with at least the following columns:

#         - 'filepath'   : full path to the *_r.mat file
#         - 'subject_id' : subject identifier extracted from filename
#         - 'label'      : label (e.g. ASD, TD, UNLABELED) [optional]

#     Processing steps:
#         1. Loads each .mat file using h5py (EEG_Resting + samplingRate).
#         2. Selects resting segments (blocks) according to `n_blocks_to_keep` and
#            `require_n_blocks`:
#               - If require_n_blocks=True: skip files with < n_blocks_to_keep blocks.
#               - If require_n_blocks=False: keep up to n_blocks_to_keep blocks.
#         3. Converts each kept segment to an MNE Raw object.
#         4. Concatenates segments, applies a standard montage.
#         5. Saves one FIF per subject into label-based subfolders.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame produced by `scan_eeg_directory` with backend="h5"
#         (or an equivalent subset). Must contain 'filepath' and 'subject_id',
#         and ideally 'label'.

#     out_dir : str or Path
#         Root output directory. Subfolders will be created per label
#         (e.g., ASD, TD, UNLABELED).

#     montage_name : str, optional
#         Name of an MNE standard montage (e.g. "GSN-HydroCel-128").
#         Must be provided and valid.

#     overwrite : bool
#         Whether to overwrite existing FIF files if they already exist.

#     n_blocks_to_keep : int
#         Target number of resting segments (blocks) to retain from EEG_Resting.

#         - If require_n_blocks=True, this is a minimum requirement: files with fewer
#           than `n_blocks_to_keep` blocks are skipped.
#         - If require_n_blocks=False, this is a maximum: files with fewer blocks are
#           still processed using all available blocks.

#     require_n_blocks : bool
#         If True, enforce that each file must contain at least `n_blocks_to_keep`
#         blocks; otherwise skip the file (and report it). If False, process files
#         even when fewer than `n_blocks_to_keep` blocks exist.

#     Returns
#     -------
#     None
#         Writes FIF files to disk and prints a summary of edge cases.

#     Notes
#     -----
#     Expected MATLAB structure:
#         - Dataset : EEG_Resting   (segments × time × channels)
#         - Dataset : samplingRate  (scalar or 1×1 matrix)

#     This function does *not* do any folder scanning or metadata merging; that is
#     handled upstream by `scan_eeg_directory`.
#     """

#     # ------------------------------------------------------------------
#     # 0. Basic validation of df and required columns
#     # ------------------------------------------------------------------
#     required_cols = {"filepath", "subject_id"}
#     missing_cols = required_cols - set(df.columns)
#     if missing_cols:
#         raise ValueError(
#             f"[convert_abcct_from_scan] df is missing required columns: {sorted(missing_cols)}"
#         )

#     # 'label' is optional, but useful; if missing, we synthesize UNLABELED.
#     if "label" not in df.columns:
#         df = df.copy()
#         df["label"] = "UNLABELED"

#     # ------------------------------------------------------------------
#     # 1. Prepare output directory and montage
#     # ------------------------------------------------------------------
#     out_dir_path = Path(out_dir).expanduser().resolve()
#     out_dir_path.mkdir(parents=True, exist_ok=True)

#     if montage_name is None:
#         raise ValueError(
#             "[convert_abcct_from_scan] You must provide montage_name "
#             "(e.g., 'GSN-HydroCel-128')."
#         )

#     try:
#         montage = mne.channels.make_standard_montage(montage_name)
#     except Exception as exc:
#         raise ValueError(
#             f"[convert_abcct_from_scan] Invalid montage '{montage_name}'. "
#             f"Check available standard montages in MNE."
#         ) from exc

#     # ------------------------------------------------------------------
#     # 2. Track edge cases for reporting
#     # ------------------------------------------------------------------
#     failed: list[tuple[str, str]] = []
#     insufficient_blocks: list[tuple[str, int]] = []  # (filename, n_found)
#     missing_label_files: list[str] = []

#     # ------------------------------------------------------------------
#     # 3. Main conversion loop
#     # ------------------------------------------------------------------
#     for row in tqdm(df.itertuples(), total=len(df), desc="Converting ABC-CT (from scan)", unit="file"):
#         mat_path = Path(row.filepath)
#         subj_id = str(row.subject_id)

#         # Handle label; default to UNLABELED if NaN or empty
#         label = getattr(row, "label", "UNLABELED")
#         if pd.isna(label) or str(label).strip() == "":
#             label = "UNLABELED"
#             missing_label_files.append(mat_path.name)

#         label = str(label).strip()

#         # Destination directory: one subfolder per label
#         label_dir = out_dir_path / label
#         label_dir.mkdir(parents=True, exist_ok=True)

#         out_path = label_dir / f"{subj_id}_eeg.fif"

#         # Skip if FIF already exists and we don't want to overwrite
#         if out_path.exists() and not overwrite:
#             continue

#         try:
#             # -----------------------------------------------------
#             # 3a. Load MATLAB data with h5py
#             # -----------------------------------------------------
#             with h5py.File(mat_path, "r") as f:
#                 X = f["EEG_Resting"][()]      # shape: (segments, time, channels)
#                 sr = f["samplingRate"][()]    # scalar or 1×1

#             # sampling rate: handle 1×1 MATLAB matrix vs scalar
#             sfreq = float(sr[0, 0]) if getattr(sr, "shape", None) == (1, 1) else float(sr)

#             if X.ndim != 3:
#                 raise ValueError(f"Unexpected EEG_Resting ndim={X.ndim} for file {mat_path.name}")

#             n_seg, n_times, n_ch = X.shape

#             # -----------------------------------------------------
#             # 3b. Decide whether to skip based on available blocks
#             # -----------------------------------------------------
#             if n_seg < int(n_blocks_to_keep):
#                 insufficient_blocks.append((mat_path.name, n_seg))
#                 if require_n_blocks:
#                     # Strict mode: skip this file entirely.
#                     continue

#             # Determine how many blocks we will actually use:
#             # - Strict mode: exactly n_blocks_to_keep (since n_seg >= n_blocks_to_keep)
#             # - Lenient mode: up to n_blocks_to_keep
#             n_keep = int(n_blocks_to_keep) if require_n_blocks else min(n_seg, int(n_blocks_to_keep))
#             X = X[:n_keep]

#             # -----------------------------------------------------
#             # 3c. Build MNE Raw objects for each block
#             # -----------------------------------------------------
#             ch_names = [f"EEG{idx+1:03d}" for idx in range(n_ch)]
#             info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

#             raws: list[mne.io.BaseRaw] = []

#             for i in range(n_keep):
#                 seg = X[i]

#                 # Ensure data has shape (n_channels, n_times)
#                 if seg.shape == (n_ch, n_times):
#                     data = seg
#                 elif seg.shape == (n_times, n_ch):
#                     data = seg.T
#                 else:
#                     # Try to infer the channel axis if shape is odd
#                     if n_ch in seg.shape:
#                         ch_axis = int(np.argmax([dim == n_ch for dim in seg.shape]))
#                         data = np.moveaxis(seg, ch_axis, 0)
#                     else:
#                         raise ValueError(
#                             f"Cannot infer channel axis for segment {i} in file {mat_path.name}"
#                         )

#                 raw_i = mne.io.RawArray(data, info.copy(), verbose=False)

#                 # Annotate this segment as a single "rest block"
#                 duration = data.shape[1] / sfreq
#                 raw_i.set_annotations(
#                     mne.Annotations(
#                         onset=[0.0],
#                         duration=[duration],
#                         description=[f"rest_block_{i+1}"],
#                     )
#                 )
#                 raws.append(raw_i)

#             # -----------------------------------------------------
#             # 3d. Concatenate all blocks and finalize channels/montage
#             # -----------------------------------------------------
#             raw = mne.concatenate_raws(raws, on_mismatch="ignore", verbose=False)

#             # Optionally rename channels to a simpler scheme (E1, E2, ...)
#             mapping = {f"EEG{idx+1:03d}": f"E{idx+1}" for idx in range(n_ch)}
#             raw.rename_channels(mapping)

#             # Apply montage
#             raw.set_montage(montage)

#             # -----------------------------------------------------
#             # 3e. Save final FIF
#             # -----------------------------------------------------
#             raw.save(out_path.as_posix(), overwrite=overwrite, verbose=False)

#         except Exception as e:
#             failed.append((mat_path.name, f"{type(e).__name__}: {e}"))

#     # ------------------------------------------------------------------
#     # 4. Summary reporting
#     # ------------------------------------------------------------------
#     if failed:
#         print("\n[convert_abcct_from_scan] Failed to process the following files:")
#         for fname, err in failed:
#             print(f" - {fname}: {err}")

#     if insufficient_blocks:
#         label = "Insufficient blocks (skipped)" if require_n_blocks else "Fewer blocks than requested (processed anyway)"
#         print(f"\n[convert_abcct_from_scan] {label}:")
#         for fname, n_found in insufficient_blocks:
#             print(f" - {fname}: found {n_found}, requested {n_blocks_to_keep}")

#     if missing_label_files:
#         print("\n[convert_abcct_from_scan] Files with missing label (treated as UNLABELED):")
#         for fname in missing_label_files:
#             print(f" - {fname}")






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




def build_label_epoch_arrays(
    data_root: str | Path,
    base_config: dict,
    skip_dirnames: list[str] = ["UNLABELED"],
) -> Tuple[Dict[str, List[np.ndarray]], List[Dict[str, str]], Dict[str, Any]]:
    """
    Discover EEG .fif files grouped by label, preprocess them, and build per-label subject arrays.

    This function walks through a root directory structured as:
        data_root/
            ├── ASD/
            │   ├── subjA_eeg.fif
            │   └── subjB_eeg.fif
            └── TD/
                ├── subjC_eeg.fif
                └── subjD_eeg.fif

    Each subfolder name (e.g., "ASD", "TD") acts as a label class. Every `.fif` file is preprocessed
    using `preprocess_eeg(cfg)`, converted to MNE Epochs, and stored in a dictionary keyed by label.
    Metadata is collected for each subject, including its label, file path, and unique subject ID.

    Parameters
    ----------
    data_root : str | Path
        Root directory containing labeled EEG folders, each with one or more `.fif` files.
    base_config : dict
        Base preprocessing configuration passed to `preprocess_eeg()`. The key
        `"eeg_file_path"` is overwritten for each subject file.
    skip_dirnames : list[str], optional
        Folder name(s) under `data_root` to ignore when discovering label classes.
        Any directory whose name matches one of these values is skipped even if it
        contains `.fif` files. Defaults to ["UNLABELED"].
    Returns
    -------
    results : dict[str, list[mne.Epochs]]
        Dictionary of lists where each key is a label (e.g., "ASD", "TD"), and
        each value is a list of preprocessed `mne.Epochs` objects for that label.

    metadata : list[dict[str, str]]
        One entry per successfully processed subject, with:
            - "file_path" : str  → full path to the `.fif` file
            - "label"     : str  → folder name / class label
            - "subject_id": str  → derived from filename
            - "label_idx" : int  → position within its label list
            - "global_idx": int  → overall append order across all labels

        These indices allow deterministic pairing between metadata and results later on.

    eeg_info : dict[str, Any]
        Basic recording information from the first successfully processed file:
            {"sfreq": float, "ch_names": list[str]}

    Notes
    -----
    - Any failed file triggers a warning but does not halt execution.
    - The output order matches the discovery order of folders and files.
    - The returned `label_idx` and `global_idx` values ensure downstream
      functions can always align subjects correctly, even if metadata is reshuffled.
    """
    # --- Validate input directory ---
    data_root = Path(data_root)
    if not data_root.exists() or not data_root.is_dir():
        raise ValueError(f"Data root '{data_root}' does not exist or is not a directory.")


    # --- Decide which directory names to skip ---
    if skip_dirnames is None:
        skip_dirnames = ["UNLABELED"]  # default

    skip_set = {d.lower() for d in skip_dirnames}  # case-insensitive compare

    # --- Discover label folders and their .fif files ---
    label_to_files = {
        sub.name: sorted(sub.glob("*.fif"))
        for sub in sorted(data_root.iterdir())
        if sub.is_dir()
        and sub.name.lower() not in skip_set
        and list(sub.glob("*.fif"))
    }


    # # --- Discover label folders and their .fif files ---
    # label_to_files = {
    #     sub.name: sorted(sub.glob("*.fif"))
    #     for sub in sorted(data_root.iterdir())
    #     if sub.is_dir() and list(sub.glob("*.fif"))
    # }

    if not label_to_files:
        raise ValueError(f"No label folders with .fif files found under '{data_root}'.")

    # --- Initialize containers ---
    results: Dict[str, List[np.ndarray]] = defaultdict(list)  # {label: [mne.Epochs, ...]}
    metadata: List[Dict[str, str]] = []                      # [{file_path, label, subject_id, ...}, ...]
    eeg_info: Dict[str, Any] = {}                            # global EEG info (sfreq, ch_names)

    # --- Counters to preserve deterministic ordering ---
    per_label_counts = defaultdict(int)  # label → subject index within label
    global_count = 0                     # overall index across all subjects
    
    # --- Loop through each label and process its .fif files ---
    for label, files in label_to_files.items():

        for fpath in files:
            # Deep copy so we don't mutate the shared base_config
            cfg = deepcopy(base_config)
    
            # Override the path in the load_eeg step for this specific file
            for step in cfg["steps"]:
                if "load_eeg" in step:
                    step["load_eeg"]['params']['path']= str(fpath)
                    break  # assume only one load_eeg step


            # Derive subject ID from filename, e.g. NDAR123_eeg.fif → NDAR123
            subject_id = fpath.stem.replace("_eeg", "")

            #print("label = ", label)
            #print("cfg[eeg_file_path] = ", cfg["eeg_file_path"])
            #print("subject_id = ", subject_id)

            try:
                # Run the preprocessing pipeline
                state = eeg_preprocess_pipeline(cfg)
                epochs_final = state['epochs_ransac'].copy()

                # Capture EEG info from the first valid file
                if not eeg_info:
                    eeg_info = {
                        "sfreq": float(epochs_final.info["sfreq"]),
                        "ch_names": list(epochs_final.ch_names),
                    }
                else:
                    # Light consistency checks
                    sfreq_now = float(epochs_final.info["sfreq"])
                    ch_now = list(epochs_final.ch_names)
                    if abs(sfreq_now - eeg_info["sfreq"]) > 1e-6:
                        print(f"[warn] sfreq mismatch in '{fpath}': got {sfreq_now}, expected {eeg_info['sfreq']}")
                    if ch_now != eeg_info["ch_names"]:
                        print(f"[warn] ch_names mismatch in '{fpath}'. Using ch_names from first success.")

                # --- Store results and metadata ---
                results[label].append(epochs_final)
                metadata.append({
                    "file_path": str(fpath),
                    "label": label,
                    "subject_id": subject_id,
                    "label_idx": per_label_counts[label],  # within-label order
                    "global_idx": global_count,             # overall order
                })

                # Increment counters (only for successful subjects)
                per_label_counts[label] += 1
                global_count += 1

            except Exception as e:
                print(f"[warn] Skipping '{fpath}' ({label}): {e}")

    
    return dict(results), metadata, eeg_info




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

