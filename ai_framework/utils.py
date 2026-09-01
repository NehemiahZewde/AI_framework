# utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type, Mapping, Literal, Callable, MutableMapping, Hashable
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

import json
from collections import defaultdict
from copy import deepcopy
import re
import pickle

import cloudpickle
import gzip
from datetime import datetime
from sklearn.model_selection import train_test_split

from numpy.typing import NDArray
from scipy.io import loadmat

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


# ---------------------------
# Functions to save and load 
# ---------------------------
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




# ---------------------------------------------------------------------
# Save and load results from training ML model
# ---------------------------------------------------------------------

# def prepare_dataset(
#     X,
#     y,
#     feature_names,
#     target_name="target",
#     validation_size=0.2,
#     random_state=42,
#     stratify=True,
# ):
#     """
#     Prepare a machine learning dataset as pandas DataFrames.

#     This function takes a feature matrix `X`, a target vector `y`, and the
#     corresponding feature names, then combines them into a single pandas
#     DataFrame. It also splits the full dataset into a training set and a
#     validation set that can be used later to evaluate machine learning models.

#     Parameters
#     ----------
#     X : array-like of shape (n_samples, n_features)
#         The input feature matrix. Each row represents one sample, and each
#         column represents one feature.

#     y : array-like of shape (n_samples,)
#         The target values or labels corresponding to each row in `X`.

#     feature_names : array-like of shape (n_features,)
#         The names to use for the feature columns in the returned DataFrames.

#     target_name : str, default="target"
#         The name of the target column that will be added to the DataFrame.

#     validation_size : float, default=0.2
#         The proportion of the full dataset to place into the validation set.
#         For example, `0.2` means 20% validation and 80% training.

#     random_state : int, default=42
#         Controls the randomness of the train-validation split so results are
#         reproducible.

#     stratify : bool, default=True
#         If True, preserves the target class distribution in both the training
#         and validation sets. This is usually helpful for classification tasks.

#     Returns
#     -------
#     train_df : pandas.DataFrame
#         The training dataset containing feature columns and the target column.

#     validation_df : pandas.DataFrame
#         The validation dataset containing feature columns and the target column.
#     """

#     # Create a DataFrame from the feature matrix using the provided column names.
#     df = pd.DataFrame(X, columns=feature_names)

#     # Add the target values as a new column in the DataFrame.
#     df[target_name] = y

#     # Use the target column for stratification if requested.
#     stratify_values = df[target_name] if stratify else None

#     # Split the full dataset into training and validation DataFrames.
#     train_df, validation_df = train_test_split(
#         df,
#         test_size=validation_size,
#         random_state=random_state,
#         stratify=stratify_values,
#     )

#     # Reset the training DataFrame index after splitting.
#     train_df = train_df.reset_index(drop=True)

#     # Reset the validation DataFrame index after splitting.
#     validation_df = validation_df.reset_index(drop=True)

#     # Return both prepared datasets.
#     return train_df, validation_df


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
# Dynamically inspect a nested CV results
# ---------------------------------------------------------------------


def summarize_value(value):
    """
    Create a compact, readable summary of a value inside one result record.
    """

    if isinstance(value, dict):
        return f"dict, keys={list(value.keys())}"

    if isinstance(value, list):
        return f"list, len={len(value)}, example={value[:5]}"

    if isinstance(value, tuple):
        return f"tuple, len={len(value)}"

    if isinstance(value, np.ndarray):
        return f"ndarray, shape={value.shape}, dtype={value.dtype}"

    if hasattr(value, "shape"):
        return f"{type(value).__name__}, shape={value.shape}"

    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    return type(value).__name__


def inspect_nested_cv_results(all_results, display_table=True):
    """
    Dynamically inspect a nested CV results dictionary.

    Expected structure:
        all_results[model_name] = list of result dictionaries

    The function does not hard-code specific keys such as metrics,
    calibration outputs, PDP outputs, or permutation importance outputs.
    It inspects whatever keys are present.

    Returns
    -------
    summary_df : pd.DataFrame
        Rows are inspected items.
        Columns are model names.
    """

    summary = {}

    for model_name, records in all_results.items():
        model_summary = {}

        model_summary["top_level_value_type"] = type(records).__name__
        model_summary["n_records"] = len(records)

        if len(records) == 0:
            summary[model_name] = model_summary
            continue

        first_record = records[0]

        model_summary["record_type"] = type(first_record).__name__
        model_summary["n_keys_in_record"] = len(first_record)

        # Dynamically summarize every key in the first record
        for key, value in first_record.items():
            model_summary[key] = summarize_value(value)

        summary[model_name] = model_summary

    summary_df = pd.DataFrame(summary)
    summary_df.index.name = "Item"
    summary_df = summary_df.reset_index()

    if display_table:
        display(summary_df)

    return summary_df