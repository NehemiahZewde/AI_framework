# eeg_features_microstate.py
# Microstate dynamic EEG feature extraction using existing functions.


from typing import List, Optional, Tuple, Dict, Any, Union, Sequence
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import mne_features
from .utils import validate_feature_names, _normalize_epochs_input
import infomeasure as im
import neurokit2 as nk



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
    

    
def feature_microstate_coverage_per_epoch(
    labels: np.ndarray,
    k: int,
    fs: float | None = None,
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute microstate coverage (fractional occupancy) per epoch and state.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs, n_times)
        Integer label array from pycrostates segmentation.
        Values are 0..k-1 for valid states, and -1 for unlabeled samples.
    k : int
        Number of microstate classes (clusters).
    fs : float, optional
        Sampling frequency in Hz. Not used in coverage computation, but kept
        for a consistent signature with other microstate feature functions.
    state_names : list[str] | None
        Optional list of names (e.g., ['A','B','C','D']).
        If None, defaults to ['0','1',..., str(k-1)].

    Returns
    -------
    cov_feat : np.ndarray, shape (n_epochs, k)
        Fractional coverage per epoch × state (values in [0, 1]).
    col_names : list[str]
        Column names like 'MS_Cover_A', 'MS_Cover_B', ...
    df : pd.DataFrame
        Coverage table, one row per epoch.
    """
    n_epochs, n_times = labels.shape
    if state_names is None:
        state_names = [str(i) for i in range(k)]

    cov_feat = np.zeros((n_epochs, k), dtype=float)

    for e in range(n_epochs):
        lab = labels[e]
        valid = lab >= 0  # ignore unlabeled samples (-1)

        if np.any(valid):
            for i in range(k):
                cov_feat[e, i] = np.mean(lab[valid] == i)
        else:
            cov_feat[e, :] = np.nan  # all unlabeled in this epoch

    col_names = [f"MS_Cover_{name}" for name in state_names]
    df = pd.DataFrame(cov_feat, columns=col_names)

    return cov_feat, col_names, df


def feature_microstate_mean_duration_per_epoch(
    labels: np.ndarray,
    k: int,
    fs: float,
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute mean microstate segment duration (in seconds) per epoch and state.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs, n_times)
        Integer label array from pycrostates segmentation.
        Values are 0..k-1 for valid states, and -1 for unlabeled samples.
        Unlabeled samples are excluded from all calculations.
    k : int
        Number of microstate classes (clusters).
    fs : float
        Sampling frequency in Hz (samples per second).
    state_names : list[str] | None
        Optional list of state names, e.g. ['A','B','C','D'].
        If None, defaults to ['0','1',..., str(k-1)].

    Returns
    -------
    dur_feat : np.ndarray, shape (n_epochs, k)
        Mean segment duration (seconds) per epoch × state.
    col_names : list[str]
        Column names like 'MeanDur_A', 'MeanDur_B', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per state.

    Notes
    -----
    - For each epoch, this function finds contiguous runs of each state.
    - The mean run length (in samples) is divided by fs to convert to seconds.
    - Epochs with no valid (non-negative) labels are assigned NaN.
    - If a specific state does not appear in an epoch, its mean duration = 0.0.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (n_epochs, n_times); got {labels.shape}")
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency `fs` must be a positive float.")

    n_epochs, n_times = labels.shape
    if state_names is None:
        state_names = [str(i) for i in range(k)]

    dur_feat = np.zeros((n_epochs, k), dtype=float)

    for e in range(n_epochs):
        lab = labels[e]
        valid = lab >= 0  # exclude unlabeled samples (-1)
        if not np.any(valid):
            dur_feat[e, :] = np.nan
            continue

        for i in range(k):
            seg_lengths = []
            in_segment = False
            start = 0

            for t in range(n_times):
                # start of a new segment
                if lab[t] == i and not in_segment:
                    in_segment = True
                    start = t
                # end of a segment (or end of sequence)
                elif (lab[t] != i or t == n_times - 1) and in_segment:
                    end = t if lab[t] != i else t + 1
                    seg_lengths.append(end - start)
                    in_segment = False

            if seg_lengths:
                dur_feat[e, i] = np.mean(seg_lengths) / fs
            else:
                dur_feat[e, i] = 0.0  # no segment for this state

    col_names = [f"MS_MeanDur_{name}" for name in state_names]
    df = pd.DataFrame(dur_feat, columns=col_names, index=np.arange(n_epochs))

    return dur_feat, col_names, df


def feature_microstate_occurrence_rate_per_epoch(
    labels: np.ndarray,
    k: int,
    fs: float,
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute microstate occurrence rate (segments per second) per epoch and state.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs, n_times)
        Integer label array from pycrostates segmentation.
        Values are 0..k-1 for valid states, and -1 for unlabeled samples.
        Unlabeled samples are excluded from all calculations.
    k : int
        Number of microstate classes (clusters).
    fs : float
        Sampling frequency in Hz (samples per second).
    state_names : list[str] | None
        Optional list of state names, e.g. ['A','B','C','D'].
        If None, defaults to ['0','1',...].

    Returns
    -------
    rate_feat : np.ndarray, shape (n_epochs, k)
        Occurrence rate (segments/second) per epoch × state.
    col_names : list[str]
        Column names like 'OccRate_A', 'OccRate_B', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per state.

    Notes
    -----
    - A "segment" is a contiguous run of the same state label.
    - The denominator is the duration of valid samples only (labels >= 0),
      i.e., (#valid_samples / fs). If an epoch has 0 valid samples, outputs NaN.
    - If a specific state does not appear in an epoch, its rate = 0.0.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (n_epochs, n_times); got {labels.shape}")
    if fs is None or fs <= 0:
        raise ValueError("Sampling frequency `fs` must be a positive float.")

    n_epochs, n_times = labels.shape
    if state_names is None:
        state_names = [str(i) for i in range(k)]

    rate_feat = np.zeros((n_epochs, k), dtype=float)

    for e in range(n_epochs):
        lab = labels[e]
        valid = lab >= 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            rate_feat[e, :] = np.nan  # no valid duration to normalize by
            continue

        valid_duration_sec = n_valid / fs  # denominator

        for i in range(k):
            # count contiguous segments for state i
            seg_count = 0
            in_segment = False
            for t in range(n_times):
                if lab[t] == i and not in_segment:
                    in_segment = True
                    seg_count += 1
                elif lab[t] != i and in_segment:
                    in_segment = False

            rate_feat[e, i] = seg_count / valid_duration_sec if seg_count > 0 else 0.0

    col_names = [f"MS_OccRate_{name}" for name in state_names]
    df = pd.DataFrame(rate_feat, columns=col_names, index=np.arange(n_epochs))

    return rate_feat, col_names, df

def feature_microstate_ShannonEntropy_per_epoch(
    labels: np.ndarray,
    k: int,
    fs: float,
    state_names: list[str] | None = None,
):
    """
    Compute Shannon entropy (nats) of the microstate label distribution per epoch.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs, n_times)
        Microstate labels (0..k-1 valid, -1 unlabeled). Unlabeled samples ignored.
    k : int
        Number of microstate classes.
    fs : float
        Sampling frequency (not used, kept only for API consistency).
    state_names : list[str] | None
        Unused for entropy; present for signature uniformity.

    Returns
    -------
    ent_feat : np.ndarray, shape (n_epochs, 1)
        Shannon entropy (nats) per epoch.
    col_names : list[str]
        ["Entropy_nats"]
    df : pd.DataFrame
        One column per epoch.
    """

    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (n_epochs, n_times); got {labels.shape}")

    n_epochs, _ = labels.shape
    ent_feat = np.full((n_epochs, 1), np.nan, dtype=float)

    for e in range(n_epochs):
        lab = labels[e]
        valid = lab >= 0

        if not np.any(valid):
            continue  # leave NaN

        # Extract valid label sequence
        data = lab[valid].tolist()

        # Shannon entropy using infomeasure discrete estimator
        est = im.estimator(data, measure="h", approach="discrete")
        ent_feat[e, 0] = float(est.result())

    col_names = ["MS_Entropy_nats"]
    df = pd.DataFrame(ent_feat, columns=col_names, index=np.arange(n_epochs))

    return ent_feat, col_names, df

def feature_microstate_lzc_per_epoch(
    labels: np.ndarray,
    k: int,
    fs: float,
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Lempel–Ziv Complexity (LZC) per epoch using NeuroKit2.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs, n_times)
        Microstate labels (0..k-1 valid, -1 unlabeled). Unlabeled are ignored.
    k : int
        Number of microstate classes (unused here; kept for API consistency).
    fs : float
        Sampling frequency in Hz (unused here; kept for API consistency).
    state_names : list[str] | None
        Present only for signature consistency with other microstate features.

    Returns
    -------
    lzc_feat : np.ndarray, shape (n_epochs, 1)
        LZC per epoch (scalar).
    col_names : list[str]
        ["MS_LZC"].
    df : pd.DataFrame
        One-column DataFrame with LZC per epoch.

    Notes
    -----
    - NeuroKit2 internally binarizes the input sequence before computing LZC.
    - If an epoch has < 2 valid samples after removing unlabeled (-1), returns NaN.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (n_epochs, n_times); got {labels.shape}")

    n_epochs, _ = labels.shape
    lzc_feat = np.full((n_epochs, 1), np.nan, dtype=float)

    for e in range(n_epochs):
        lab = labels[e]
        lab = lab[lab >= 0]  # drop unlabeled samples
        if lab.size < 2:
            continue  # keep NaN
        # NeuroKit2 returns (value, info); value is LZC of binarized sequence
        lzc_val, _ = nk.complexity_lempelziv(lab)
        lzc_feat[e, 0] = float(lzc_val)

    col_names = ["MS_LZC"]
    df = pd.DataFrame(lzc_feat, columns=col_names, index=np.arange(n_epochs))
    return lzc_feat, col_names, df

def feature_microstate_transition_probs_per_epoch(
    labels: np.ndarray,
    k: int,
    fs: float | None,  # kept for API consistency; not used
    state_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Transition probabilities per epoch (row-stochastic over OFF-DIAGONALS), flattened.

    - Unlabeled (-1) samples are removed.
    - Repetitions are ignored (run-length encoding) so only actual switches count.
    - Diagonal self-transitions (i->i) are DROPPED; rows are renormalized over j != i.
    - If a 'from' state has no observed switches to any other state within the epoch,
      its row is left as all zeros.

    Parameters
    ----------
    labels : np.ndarray, shape (n_epochs, n_times)
        Microstate labels in 0..k-1; -1 = unlabeled.
    k : int
        Number of microstate classes.
    fs : float or None
        Sampling frequency in Hz (unused; kept for API consistency with other MS features).
    state_names : list[str] | None
        Optional names like ['A','B','C','D'] for column labels.

    Returns
    -------
    tp_feat : np.ndarray, shape (n_epochs, k*(k-1))
        Row-major flattened OFF-DIAGONAL transition probability matrices.
    col_names : list[str]
        Column names like 'MS_TP_A_to_B' (no self-transitions included).
    df : pd.DataFrame
        One row per epoch, k*(k-1) columns.
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (n_epochs, n_times); got {labels.shape}")

    n_epochs, _ = labels.shape
    if state_names is None:
        state_names = [str(i) for i in range(k)]

    # Off-diagonal column names (row-major over i, then j!=i)
    pairs = [(i, j) for i in range(k) for j in range(k) if j != i]
    col_names = [f"MS_TP_{state_names[i]}_to_{state_names[j]}" for (i, j) in pairs]

    rows = []
    for e in range(n_epochs):
        lab = labels[e]
        lab = lab[lab >= 0]  # drop unlabeled

        if lab.size < 2:
            rows.append([0.0] * (k * (k - 1)))
            continue

        # Run-length encode to keep only true switches
        rle = [lab[0]]
        for x in lab[1:]:
            if x != rle[-1]:
                rle.append(x)
        seq = np.array(rle, dtype=int)
        if seq.size < 2:
            rows.append([0.0] * (k * (k - 1)))
            continue

        # NeuroKit2 transition matrix (row-normalized)
        tm_df, _ = nk.transition_matrix(seq, adjust=True)

        # Reindex to full k×k, fill missing with 0
        tm_df = tm_df.reindex(index=range(k), columns=range(k), fill_value=0.0)

        # Zero the diagonal and renormalize each row across off-diagonals
        tm = tm_df.values.astype(float)
        np.fill_diagonal(tm, 0.0)
        row_sums = tm.sum(axis=1, keepdims=True)
        nz = row_sums.squeeze(-1) > 0
        tm[nz] = tm[nz] / row_sums[nz]

        # Flatten OFF-DIAGONALS row-major
        flat = [float(tm[i, j]) for (i, j) in pairs]
        rows.append(flat)

    tp_feat = np.array(rows, dtype=float)
    df = pd.DataFrame(tp_feat, columns=col_names, index=np.arange(n_epochs))
    return tp_feat, col_names, df


# -----------------------------------------------------------------------------
# Key → Function Mapping
# -----------------------------------------------------------------------------
FEATURE_REGISTRY_MS = {
    "Cover":      feature_microstate_coverage_per_epoch,            # MS1 — Fractional occupancy (% time in each state)
    "MeanDur":   feature_microstate_mean_duration_per_epoch,       # MS2 — Mean segment duration (s) per microstate
    "OccRate":   feature_microstate_occurrence_rate_per_epoch,     # MS3 — Segment occurrence rate (segments/s)
    "Entropy":    feature_microstate_ShannonEntropy_per_epoch,      # MS4 — Shannon entropy (nats)
    "LZC":        feature_microstate_lzc_per_epoch,                 # MS5 — Lempel–Ziv complexity (temporal complexity)
    "MS_TP":      feature_microstate_transition_probs_per_epoch,    # MS6 — Transition probabilities (off-diagonal, normalized)
}

FEATURE_DOCS_MS = {
    "Cover":   "Fractional occupancy of each microstate per epoch.",
    "MeanDur": "Mean duration (s) of each state’s segments per epoch.",
    "OccRate": "Occurrence rate (segments/s) per state.",
    "Entropy": "Shannon entropy (nats) of label distribution.",
    "LZC":     "Lempel–Ziv complexity of the label sequence.",
    "MS_TP":   "Off-diagonal transition probabilities (row-normalized).",
}


def extract_microstate_features(
    ms_out: Dict[str, Any],
    metadata: List[Dict[str, Any]],
    features: List[str] = ['Cover', 'MeanDur', 'OccRate', 'Entropy', 'LZC', 'MS_TP'],
    feature_kwargs: Optional[Dict[str, dict]] = None,
) -> Tuple[List[np.ndarray], Dict[str, List[str]], Dict[str, pd.DataFrame]]:
    """
    Per-subject microstate epoch features (no stacking), aligned with metadata order.

    IMPORTANT
    ---------
    Unlike the earlier version, this DOES NOT assume all subjects share the same
    number of microstates k. Therefore, different subjects can have different
    feature dimensionalities and column names.

    Returns
    -------
    features_list : list[np.ndarray]
        One array per subject, in the same order as `metadata`.
        Each array has shape (n_epochs_i, F_i), where F_i may differ by subject.

    colnames_by_subject : dict[str, list[str]]
        Mapping from "label_subjectID" (e.g. 'ASD_NDARBX669DJY') to that subject's
        feature column names.

    dfs_dict : dict[str, pd.DataFrame]
        Mapping from "label_subjectID" to a per-subject feature DataFrame.
    """
    feature_kwargs = feature_kwargs or {}
    artifacts = ms_out["artifacts"]

    features_list: List[np.ndarray] = []
    colnames_by_subject: Dict[str, List[str]] = {}
    dfs_dict: Dict[str, pd.DataFrame] = {}

    for meta in metadata:
        label = meta["label"]
        subject_id = meta["subject_id"]
        key = (label, subject_id)
        subj_key = f"{label}_{subject_id}"

        art = artifacts[key]
        labels = art["labels"]
        k = int(art["k"])
        fs = float(art["fs"])
        state_names = art.get("state_names")

        arrays_i: List[np.ndarray] = []
        cols_i: List[str] = []
        dfs_i: List[pd.DataFrame] = []

        for name in features:
            if name not in FEATURE_REGISTRY_MS:
                raise ValueError(f"Unknown microstate feature: '{name}'")

            fn = FEATURE_REGISTRY_MS[name]
            kwargs = feature_kwargs.get(name, {})

            # Note: fn signature is (labels, k, fs, state_names, **kwargs)
            arr, cols, df = fn(labels, k, fs, state_names, **kwargs)

            arrays_i.append(arr)
            cols_i.extend(cols)
            dfs_i.append(df)

        # Concatenate features for this subject
        X_i = np.concatenate(arrays_i, axis=1) if len(arrays_i) > 1 else arrays_i[0]
        df_i = pd.concat(dfs_i, axis=1)

        features_list.append(X_i)
        dfs_dict[subj_key] = df_i
        colnames_by_subject[subj_key] = cols_i

    return features_list, colnames_by_subject, dfs_dict



# def extract_microstate_features(
#     ms_out: Dict[str, Any],
#     metadata: List[Dict[str, Any]],
#     features: List[str] = ['Cover', 'MeanDur', 'OccRate', 'Entropy', 'LZC', 'MS_TP'],
#     feature_kwargs: Optional[Dict[str, dict]] = None,
# ) -> Tuple[List[np.ndarray], List[str], Dict[str, pd.DataFrame]]:
#     """
#     Per-subject microstate epoch features (no stacking), aligned with metadata order.

#     Returns
#     -------
#     features_list : list[np.ndarray]
#         One array per subject, shape (n_epochs_i, F_total).
#     colnames : list[str]
#         Shared column names (same across subjects).
#     dfs_dict : dict[str, pd.DataFrame]
#         Dictionary keyed by 'label_subjectID' (e.g., 'ASD_NDARBX669DJY'),
#         with each value a per-subject feature DataFrame.
#     """


#     feature_kwargs = feature_kwargs or {}
#     artifacts = ms_out["artifacts"]

#     features_list: List[np.ndarray] = []
#     colnames: List[str] = []
#     dfs_dict: Dict[str, pd.DataFrame] = {}

#     for idx, meta in enumerate(metadata):
#         label = meta["label"]
#         subject_id = meta["subject_id"]
#         key = (label, subject_id)

#         art = artifacts[key]
#         labels = art["labels"]
#         k = int(art["k"])
#         fs = float(art["fs"])
#         state_names = art.get("state_names")

#         arrays_i, cols_i, dfs_i = [], [], []

#         for name in features:
#             if name not in FEATURE_REGISTRY_MS:
#                 raise ValueError(f"Unknown microstate feature: '{name}'")

#             fn = FEATURE_REGISTRY_MS[name]
#             kwargs = feature_kwargs.get(name, {})
            
#             arr, cols, df = fn(labels, k, fs, state_names, **kwargs)


#             arrays_i.append(arr)
#             cols_i.extend(cols)
#             dfs_i.append(df)

#         X_i = np.concatenate(arrays_i, axis=1) if len(arrays_i) > 1 else arrays_i[0]
#         df_i = pd.concat(dfs_i, axis=1)

#         if idx == 0:
#             colnames = cols_i
#         else:
#             if cols_i != colnames:
#                 raise ValueError(
#                     f"Microstate feature columns mismatch at subject {key}. "
#                     f"Expected {len(colnames)}, got {len(cols_i)}."
#                 )

#         features_list.append(X_i)
#         dfs_dict[f"{label}_{subject_id}"] = df_i

#     return features_list, colnames, dfs_dict




    
