# eeg_features_time.py
# Time-domain EEG feature extraction using existing functions.

from typing import List, Optional, Tuple, Dict, Any, Union, Sequence
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import mne_features
from .utils import validate_feature_names, _normalize_epochs_input


### Time domain features
# - Mean
# - Peak-to-Peak Amplitude
# - Skewness
# - Kurtosis
# - Root-Mean Squared Value 
# - Hjorth Parameter: Mobility 
# - Quantile 
# - Hjorth Parameter: Complexity
# - Variance
# - Decorrelation Time
# - Number of zero-crossings



# ------------------------ Your functions (unchanged behavior) ------------------------

def feature_mean_per_epoch(epochs) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain MEAN for each epoch x channel.

    Returns
    -------
    mean_feat : np.ndarray, shape (n_epochs, n_channels)
        Mean value of the signal in each epoch & channel.
    col_names : list of str
        Column names like 'T2_mean__<chan>'.
    df : pd.DataFrame
        Same data as `mean_feat`, but as a DataFrame with columns per channel.
    """
    # MNE Epochs -> ndarray (n_epochs, n_channels, n_times)
    X = epochs.get_data()  # float64 by default
    # robust guard: replace NaNs/inf if any slipped through
    X = np.nan_to_num(X, copy=False)

    # axis=-1 is time; mean over samples inside each epoch
    mean_feat = X.mean(axis=-1)  # (n_epochs, n_channels)

    ch_names = epochs.ch_names
    col_names = [f"T2_mean__{ch}" for ch in ch_names]

    df = pd.DataFrame(mean_feat, columns=col_names, index=np.arange(mean_feat.shape[0]))
    return mean_feat, col_names, df

def feature_std_per_epoch(epochs):
    """
    Compute the time-domain standard deviation feature (T1) for each epoch and channel.

    Each epoch is a 2D array (channels × time points).
    This function calculates the standard deviation of signal amplitude over time for each channel
    (i.e., row-wise std across time samples).

    Returns
    -------
    std_feat : np.ndarray, shape (n_epochs, n_channels)
        Standard deviation of the signal in each epoch & channel.
    col_names : list of str
        Column names like 'T1_std__<chan>'.
    df : pd.DataFrame
        DataFrame with one column per channel and one row per epoch.
    """

    # Extract EEG data: (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)

    # --- Compute standard deviation per channel (row-wise std across time) ---
    # axis=-1 corresponds to the time dimension.
    # We calculate std over time to get one std value per channel in each epoch.
    std_feat = X.std(axis=-1, ddof=0)

    # Label columns
    ch_names = epochs.ch_names
    col_names = [f"T1_std__{ch}" for ch in ch_names]

    # Convert to DataFrame for convenience
    df = pd.DataFrame(std_feat, columns=col_names)
    return std_feat, col_names, df





def feature_var_per_epoch(epochs):
    """
    Compute the time-domain variance feature (T10) for each epoch and channel.

    Each epoch is a 2D array (channels × time points).
    This function calculates the variance of signal amplitude over time for each channel
    (i.e., row-wise variance across time samples).

    Returns
    -------
    var_feat : np.ndarray, shape (n_epochs, n_channels)
        Variance of the signal in each epoch & channel.
    col_names : list of str
        Column names like 'T10_var__<chan>'.
    df : pd.DataFrame
        DataFrame with one column per channel and one row per epoch.
    """

    # Extract EEG data: (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)

    # --- Compute variance per channel (row-wise variance across time) ---
    # axis=-1 corresponds to the time dimension.
    # We calculate variance over time to get one variance value per channel in each epoch.
    var_feat = X.var(axis=-1, ddof=0)

    # Label columns
    ch_names = epochs.ch_names
    col_names = [f"T10_var__{ch}" for ch in ch_names]

    # Convert to DataFrame for convenience
    df = pd.DataFrame(var_feat, columns=col_names)
    return var_feat, col_names, df

def feature_rms_per_epoch(epochs):
    """
    Compute the time-domain RMS (T6) for each epoch and channel using mne-features.

    Returns
    -------
    rms_feat : np.ndarray, shape (n_epochs, n_channels)
        RMS value per epoch × channel.
    col_names : list[str]
        Column names like 'T6_rms__<chan>'.
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # Extract EEG data
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    X = np.nan_to_num(X, copy=False)
    n_epochs, n_channels, _ = X.shape
    ch_names = epochs.ch_names

    # --- Compute RMS per epoch (expects (n_channels, n_times)) ---
    rms_list = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        rms_val = mne_features.univariate.compute_rms(epoch_data)  # (n_channels,)
        rms_list.append(rms_val)

    # Stack results
    rms_feat = np.vstack(rms_list)  # (n_epochs, n_channels)

    # Column names and DataFrame
    col_names = [f"T6_rms__{ch}" for ch in ch_names]
    df = pd.DataFrame(rms_feat, columns=col_names)

    return rms_feat, col_names, df


def feature_ptp_per_epoch(epochs):
    """
    Compute the time-domain peak-to-peak amplitude feature (T3) for each epoch and channel.

    Each epoch is a 2D array (channels × time points).
    This function calculates the max-minus-min amplitude over time for each channel
    (i.e., row-wise peak-to-peak across time samples).

    Returns
    -------
    ptp_feat : np.ndarray, shape (n_epochs, n_channels)
        Peak-to-peak value of the signal in each epoch & channel.
    col_names : list of str
        Column names like 'T3_ptp__<chan>'.
    df : pd.DataFrame
        DataFrame with one column per channel and one row per epoch.
    """

    # Extract EEG data: (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)

    # --- Compute peak-to-peak per channel (row-wise across time) ---
    # axis=-1 corresponds to the time dimension.
    ptp_feat = np.ptp(X, axis=-1)

    # Label columns
    ch_names = epochs.ch_names
    col_names = [f"T3_ptp__{ch}" for ch in ch_names]

    # Convert to DataFrame for convenience
    df = pd.DataFrame(ptp_feat, columns=col_names)
    return ptp_feat, col_names, df


def feature_skew_per_epoch(epochs, bias=True, nan_policy="propagate"):
    """
    Compute the time-domain skewness feature (T4) for each epoch and channel.

    Each epoch is a 2D array (channels × time points).
    This function calculates the skewness of signal amplitude over time for each channel
    using scipy.stats.skew (i.e., row-wise skew across time samples).

    Parameters
    ----------
    bias : bool
        If False, then the calculations are corrected for statistical bias (sample estimator).
        If True, returns the population standardized moment (no bias correction).
    nan_policy : {'propagate', 'raise', 'omit'}
        Defines how to handle when input contains NaN.

    Returns
    -------
    skew_feat : np.ndarray, shape (n_epochs, n_channels)
        Skewness of the signal in each epoch & channel.
    col_names : list of str
        Column names like 'T4_skew__<chan>'.
    df : pd.DataFrame
        DataFrame with one column per channel and one row per epoch.
    """

    # Extract EEG data: (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)

    # --- Compute skewness per channel (row-wise across time) ---
    # axis=-1 corresponds to the time dimension.
    skew_feat = skew(X, axis=-1, bias=bias, nan_policy=nan_policy)

    # Label columns
    ch_names = epochs.ch_names
    col_names = [f"T4_skew__{ch}" for ch in ch_names]

    # Convert to DataFrame for convenience
    df = pd.DataFrame(skew_feat, columns=col_names)
    return skew_feat, col_names, df



def feature_kurt_per_epoch(epochs, fisher=False, bias=True, nan_policy="propagate"):
    """
    Compute the time-domain kurtosis feature (T5) for each epoch and channel.

    Each epoch is a 2D array (channels × time points).
    This function calculates the kurtosis of signal amplitude over time for each channel
    using scipy.stats.kurtosis (i.e., row-wise kurtosis across time samples).

    Parameters
    ----------
    fisher : bool
        If True, Fisher’s definition is used (normal ==> 0.0; "excess" kurtosis).
        If False, the "Pearson" definition is used (normal ==> 3.0). Many papers
        report plain kurtosis, so fisher=False is typically what you want.
    bias : bool
        If False, the calculations are corrected for statistical bias (sample estimator).
        If True, returns the population standardized moment (no bias correction).
    nan_policy : {'propagate', 'raise', 'omit'}
        Defines how to handle when input contains NaN.

    Returns
    -------
    kurt_feat : np.ndarray, shape (n_epochs, n_channels)
        Kurtosis of the signal in each epoch & channel.
    col_names : list of str
        Column names like 'T5_kurt__<chan>'.
    df : pd.DataFrame
        DataFrame with one column per channel and one row per epoch.
    """

    # Extract EEG data: (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)

    # --- Compute kurtosis per channel (row-wise across time) ---
    # axis=-1 corresponds to the time dimension.
    kurt_feat = kurtosis(X, axis=-1, fisher=fisher, bias=bias, nan_policy=nan_policy)

    # Label columns
    ch_names = epochs.ch_names
    col_names = [f"T5_kurt__{ch}" for ch in ch_names]

    # Convert to DataFrame for convenience
    df = pd.DataFrame(kurt_feat, columns=col_names)
    return kurt_feat, col_names, df



def feature_zcr_per_epoch(epochs):
    """
    Compute the time-domain number of zero-crossings (T12) for each epoch and channel.

    Each epoch is a 2D array (channels × time points).
    This function counts the number of times the signal changes sign across time for each channel
    (i.e., row-wise zero-crossings across time samples).

    Definition
    ----------
    A zero-crossing is counted when consecutive samples have opposite signs:
        X[..., t] * X[..., t+1] < 0
    (Transitions that include an exact zero are not counted here; this is a common, robust choice.)

    Returns
    -------
    zcr_feat : np.ndarray, shape (n_epochs, n_channels)
        Number of zero-crossings in each epoch & channel.
    col_names : list of str
        Column names like 'T12_zcr__<chan>'.
    df : pd.DataFrame
        DataFrame with one column per channel and one row per epoch.
    """

    # Extract EEG data: (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)

    # --- Compute zero-crossings per channel (row-wise across time) ---
    # Count sign changes between consecutive samples (ignore exact zeros by strict < 0 test on product).
    prod = X[..., :-1] * X[..., 1:]                 # (n_epochs, n_channels, n_times-1)
    zcr_feat = (prod < 0).sum(axis=-1).astype(int)  # (n_epochs, n_channels)

    # Label columns
    ch_names = epochs.ch_names
    col_names = [f"T12_zcr__{ch}" for ch in ch_names]

    # Convert to DataFrame for convenience
    df = pd.DataFrame(zcr_feat, columns=col_names)
    return zcr_feat, col_names, df



def feature_quantile_per_epoch(epochs, q=(0.25, 0.50, 0.75)):
    """
    Compute time-domain quantiles (T8) for each epoch & channel.

    Each epoch is a 2D array (channels × time points).  This function computes
    the specified quantiles of signal amplitude across time samples within
    each channel (i.e., row-wise quantiles) and returns them concatenated
    in channel-major order.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs object.
    q : tuple/list of float in [0, 1]
        Quantile levels to compute (e.g., (0.25, 0.5, 0.75)).

    Returns
    -------
    q_feat_all : np.ndarray, shape (n_epochs, n_channels * len(q))
        Quantile values concatenated in *channel-major* order:
        [ch1_q25, ch1_q50, ch1_q75, ch2_q25, ch2_q50, ch2_q75, ...]
    col_names_all : list of str
        Column names like 'T8_q25__Fp1', 'T8_q50__Fp1', ... matching data order.
    df_all : pd.DataFrame
        Same data as `q_feat_all`, one column per (channel × quantile).
    """

    # ----------------------------------------------------------------------
    # 1) Extract EEG data: shape (n_epochs, n_channels, n_times)
    # ----------------------------------------------------------------------
    X = epochs.get_data()
    # Replace any NaN/inf with finite values to prevent NaN propagation
    X = np.nan_to_num(X, copy=False)

    # ----------------------------------------------------------------------
    # 2) Compute quantiles across time for each (epoch, channel)
    # ----------------------------------------------------------------------
    q = tuple(q)
    # np.quantile over axis=-1 gives shape (len(q), n_epochs, n_channels)
    q_vals = np.quantile(X, q, axis=-1)

    # Move quantile axis to the end → shape (n_epochs, n_channels, len(q))
    q_vals = np.moveaxis(q_vals, 0, -1)

    # ----------------------------------------------------------------------
    # 3) Flatten quantiles per channel (channel-major order)
    # ----------------------------------------------------------------------
    n_epochs, n_channels, n_q = q_vals.shape
    q_feat_all = q_vals.reshape(n_epochs, n_channels * n_q)
    #   → [ch1_q25, ch1_q50, ch1_q75, ch2_q25, ch2_q50, ch2_q75, ...]

    # ----------------------------------------------------------------------
    # 4) Build column names in the same (channel-major) order
    # ----------------------------------------------------------------------
    ch_names = epochs.ch_names
    q_perc = [int(round(100 * qi)) for qi in q]  # convert to percent, e.g., 25, 50, 75
    col_names_all = []
    for ch in ch_names:
        for qi in q_perc:
            col_names_all.append(f"T8_q{qi}__{ch}")

    # ----------------------------------------------------------------------
    # 5) Convert to DataFrame for convenience
    # ----------------------------------------------------------------------
    df_all = pd.DataFrame(q_feat_all, columns=col_names_all)

    return q_feat_all, col_names_all, df_all

def feature_hjorth_mobility_per_epoch(epochs):
    """
    Compute Hjorth Mobility (T7) for each epoch and channel using mne-features.

    Returns
    -------
    hj_mob : np.ndarray, shape (n_epochs, n_channels)
        Hjorth mobility per epoch ×build_label_epoch_arrays  channel.
    col_names : list[str]
        Column names like 'T7_hj_mob__<chan>'.
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # Extract data
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    X = np.nan_to_num(X, copy=False)
    n_epochs, n_channels, _ = X.shape
    ch_names = epochs.ch_names

    # Compute per epoch
    hj_mob_list = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hj_mob = mne_features.univariate.compute_hjorth_mobility(epoch_data)
        hj_mob_list.append(hj_mob)

    hj_mob_feat = np.vstack(hj_mob_list)  # (n_epochs, n_channels)

    # Column names and DataFrame
    col_names = [f"T7_hj_mob__{ch}" for ch in ch_names]
    df = pd.DataFrame(hj_mob_feat, columns=col_names)

    return hj_mob_feat, col_names, df

def feature_hjorth_complexity_per_epoch(epochs):
    """
    Compute Hjorth Complexity (T9) for each epoch and channel using mne-features.

    Returns
    -------
    hj_com : np.ndarray, shape (n_epochs, n_channels)
        Hjorth complexity per epoch × channel.
    col_names : list[str]
        Column names like 'T9_hj_com__<chan>'.
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # Extract data
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    X = np.nan_to_num(X, copy=False)
    n_epochs, n_channels, _ = X.shape
    ch_names = epochs.ch_names

    # Compute per epoch
    hj_com_list = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hj_com = mne_features.univariate.compute_hjorth_complexity(epoch_data)
        hj_com_list.append(hj_com)

    hj_com_feat = np.vstack(hj_com_list)  # (n_epochs, n_channels)

    # Column names and DataFrame
    col_names = [f"T9_hj_com__{ch}" for ch in ch_names]
    df = pd.DataFrame(hj_com_feat, columns=col_names)

    return hj_com_feat, col_names, df


def feature_decorr_time_per_epoch(epochs, sfreq=None):
    """
    Compute Decorrelation Time (T11) for each epoch and channel using mne-features.

    Returns
    -------
    dt_feat : np.ndarray, shape (n_epochs, n_channels)
        Decorrelation time per epoch × channel (seconds).
    col_names : list[str]
        Column names like 'T11_decorr_time__<chan>'.
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # (n_epochs, n_channels, n_times)
    X = epochs.get_data()
    X = np.nan_to_num(X, copy=False)
    n_epochs, n_channels, _ = X.shape
    ch_names = epochs.ch_names

    if sfreq is None:
        sfreq = float(epochs.info['sfreq'])

    # per-epoch loop: expects (n_channels, n_times) -> (n_channels,)
    rows = []
    for i in range(n_epochs):
        epoch_dt = mne_features.univariate.compute_decorr_time(sfreq, X[i]).astype(float)
        rows.append(epoch_dt)

    dt_feat = np.vstack(rows)  # (n_epochs, n_channels)

    col_names = [f"T11_decorr_time__{ch}" for ch in ch_names]
    df = pd.DataFrame(dt_feat, columns=col_names)
    return dt_feat, col_names, df

def concat_feature_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Column-wise concatenate multiple feature DataFrames, aligning by epoch index.
    """
    if not dfs:
        return pd.DataFrame()
    # ensure consistent row indexing
    base_index = dfs[0].index
    dfs = [d.reindex(base_index) for d in dfs]
    return pd.concat(dfs, axis=1)


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




# -----------------------------------------------------------------------------
# FEATURE_REGISTRY_TIME
# -----------------------------------------------------------------------------
# This registry acts as a central lookup table that maps short feature names
# (used in the extract_time_domain_features() call) to their corresponding
# implementation functions.
#
# ✅ Why it’s useful:
#    - Makes it easy to dynamically select which time-domain features to extract
#      without editing the core logic.
#    - Each key corresponds to a specific EEG feature function defined above.
#    - The registry allows clean, modular control of which features are computed.
#
# Example usage:
#     X, cols, df = extract_time_domain_features(epochs, features=("mean", "rms"))
#
#     → will automatically call feature_mean_per_epoch() and feature_rms_per_epoch()
# -----------------------------------------------------------------------------

FEATURE_REGISTRY_TIME = {
    "std":        feature_std_per_epoch,              # T1 — Standard deviation of amplitude over time
    "mean":       feature_mean_per_epoch,             # T2 — Mean signal amplitude per epoch × channel
    "ptp":        feature_ptp_per_epoch,              # T3 — Peak-to-peak amplitude (max − min)
    "skew":       feature_skew_per_epoch,             # T4 — Skewness (asymmetry of amplitude distribution)
    "kurt":       feature_kurt_per_epoch,             # T5 — Kurtosis (tailedness / peakedness)
    "rms":        feature_rms_per_epoch,              # T6 — Root-mean-square (signal power measure)    
    "hj_mob":     feature_hjorth_mobility_per_epoch,  # T7 — Hjorth Mobility (ratio of signal derivative stds)
    "quantile":   feature_quantile_per_epoch,         # T8 — Quantile statistics (e.g., 25/50/75th percentiles)
    "hj_com":     feature_hjorth_complexity_per_epoch,# T9 — Hjorth Complexity (shape complexity)
    "var":        feature_var_per_epoch,              # T10 — Variance of amplitude over time
    "decorr_time":feature_decorr_time_per_epoch,      # T11 — Decorrelation time (autocorrelation decay)
    "zcr":        feature_zcr_per_epoch,              # T12 — Zero-crossing rate (sign-change count)
}

# eeg_features_time.py
FEATURE_DOCS_TIME = {
    "std":         "Standard deviation of EEG amplitude over time (per channel).",
    "mean":        "Mean EEG amplitude within each epoch (per channel).",
    "ptp":         "Peak-to-peak amplitude (max − min) per epoch per channel.",
    "skew":        "Skewness of the amplitude distribution (signal asymmetry).",
    "kurt":        "Kurtosis of amplitude distribution (tailedness/peakedness).",
    "rms":         "Root-mean-square amplitude (signal energy measure).",
    "hj_mob":      "Hjorth Mobility — ratio of std of first derivative to signal std.",
    "quantile":    "Selected quantile statistics (e.g., 25th, 50th, 75th percentiles).",
    "hj_com":      "Hjorth Complexity — shape complexity relative to a sine wave.",
    "var":         "Variance of amplitude across time (per epoch per channel).",
    "decorr_time": "Decorrelation time — lag where autocorrelation first drops below 1/e.",
    "zcr":         "Zero-crossing rate — frequency of sign changes in the signal.",
}



def extract_time_domain_features(
    epochs,
    features=("mean", "std"),
    **feature_kwargs
):
    """
    Run selected time-domain feature functions and concatenate results.

    Parameters
    ----------
    epochs : mne.Epochs
    features : tuple/list of str
        Names looked up in FEATURE_REGISTRY_TIME.
    feature_kwargs : dict
        Optional per-feature kwargs, by name, e.g.:
        feature_kwargs = {
            "std": {"ddof": 0},
            "quantile": {"q": [0.25, 0.75]},
        }

    Returns
    -------
    X_all : np.ndarray, shape (n_epochs, sum(n_channels per feature))
    colnames_all : list[str]
    df_all : pd.DataFrame
    """
    # --- Initialize collectors for arrays, names, and DataFrames ---
    arrays, names, dfs = [], [], []
    
    # validate and normalize requested codes
    features = validate_feature_names(features, registry=FEATURE_REGISTRY_TIME, domain="Time")
    
    for name in features:
        if name not in FEATURE_REGISTRY_TIME:
            raise ValueError(f"Unknown time feature: '{name}'")
        fn = FEATURE_REGISTRY_TIME[name]
        kwargs = feature_kwargs.get(name, {})
        arr, cols, df = fn(epochs, **kwargs) if kwargs else fn(epochs)
        
        arrays.append(arr)
        names.extend(cols)
        dfs.append(df)

    # concatenate along feature axis
    X_all = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]
    df_all = concat_feature_dfs(dfs)
    return X_all, names, df_all






def feature_std_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain STANDARD DEVIATION for each epoch × channel.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, optional
        Unused by 'std'; present for consistent signature across features.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    std_feat : np.ndarray, shape (n_epochs, n_channels)
        Standard deviation of the signal in each epoch & channel.
    col_names : list[str]
        Column names like 'T1_std__<chan>'.
    df : pd.DataFrame
        DataFrame version of `std_feat` with one column per channel.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    # Compute per-epoch, per-channel standard deviation (across time)
    std_feat = X.std(axis=-1, ddof=0)  # (n_epochs, n_channels)

    # Build column names and DataFrame
    col_names = [f"std_{ch}" for ch in ch_names]
    df = pd.DataFrame(std_feat, columns=col_names, index=np.arange(std_feat.shape[0]))

    return std_feat, col_names, df


def feature_mean_per_epoch(
    X: np.ndarray,  
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain MEAN for each epoch × channel.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, optional
        Unused by 'mean'; present for consistent signature across features.
    ch_names : list[str], required
        Channel names (len == n_channels).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    mean_feat = X.mean(axis=-1)  # (n_epochs, n_channels)
    col_names = [f"mean_{ch}" for ch in ch_names]
    df = pd.DataFrame(mean_feat, columns=col_names, index=np.arange(mean_feat.shape[0]))
    return mean_feat, col_names, df

def feature_ptp_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain PEAK-TO-PEAK for each epoch × channel.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    ptp_feat = np.ptp(X, axis=-1)  # (n_epochs, n_channels)

    col_names = [f"ptp_{ch}" for ch in ch_names]
    df = pd.DataFrame(ptp_feat, columns=col_names, index=np.arange(ptp_feat.shape[0]))
    return ptp_feat, col_names, df

def feature_skew_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain SKEWNESS for each epoch × channel.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    # Compute skewness per epoch × channel
    skew_feat = skew(X, axis=-1, bias=bias, nan_policy=nan_policy)

    col_names = [f"skew_{ch}" for ch in ch_names]
    df = pd.DataFrame(skew_feat, columns=col_names, index=np.arange(skew_feat.shape[0]))
    return skew_feat, col_names, df

def feature_var_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain VARIANCE for each epoch × channel.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, optional
        Unused by 'var'; present for consistent signature across features.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    var_feat : np.ndarray, shape (n_epochs, n_channels)
        Variance of the signal in each epoch & channel.
    col_names : list[str]
        Column names like 'T10_var__<chan>'.
    df : pd.DataFrame
        DataFrame version of `var_feat` with one column per channel.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    # Compute per-epoch, per-channel variance (across time)
    var_feat = X.var(axis=-1, ddof=0)  # (n_epochs, n_channels)

    # Build labeled DataFrame
    col_names = [f"var_{ch}" for ch in ch_names]
    df = pd.DataFrame(var_feat, columns=col_names, index=np.arange(var_feat.shape[0]))

    return var_feat, col_names, df


def feature_rms_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain ROOT MEAN SQUARE for each epoch × channel.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, optional
        Unused by 'rms'; present for consistent signature across features.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    rms_feat : np.ndarray, shape (n_epochs, n_channels)
        RMS value per epoch & channel.
    col_names : list[str]
        Column names like 'T6_rms__<chan>'.
    df : pd.DataFrame
        DataFrame version of `rms_feat` with one column per channel.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape
    rms_feat = np.zeros((n_epochs, n_channels), dtype=float)

    # Compute RMS for each epoch and channel
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        rms_feat[i, :] = mne_features.univariate.compute_rms(epoch_data)

    # Build labeled DataFrame
    col_names = [f"rms_{ch}" for ch in ch_names]
    df = pd.DataFrame(rms_feat, columns=col_names, index=np.arange(rms_feat.shape[0]))

    return rms_feat, col_names, df

def feature_kurt_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    fisher: bool = False,
    bias: bool = True,
    nan_policy: str = "propagate",
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain KURTOSIS for each epoch × channel.
    """
    from scipy.stats import kurtosis

    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    kurt_feat = kurtosis(X, axis=-1, fisher=fisher, bias=bias, nan_policy=nan_policy)  # (n_epochs, n_channels)

    col_names = [f"kurt_{ch}" for ch in ch_names]
    df = pd.DataFrame(kurt_feat, columns=col_names, index=np.arange(kurt_feat.shape[0]))
    return kurt_feat, col_names, df

def feature_zcr_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain ZERO-CROSSING COUNT for each epoch × channel.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    # Count sign changes between consecutive samples; ignore transitions through exact zero
    prod = X[..., :-1] * X[..., 1:]                 # (n_epochs, n_channels, n_times-1)
    zcr_feat = (prod < 0).sum(axis=-1).astype(int)  # (n_epochs, n_channels)

    col_names = [f"zcr_{ch}" for ch in ch_names]
    df = pd.DataFrame(zcr_feat, columns=col_names, index=np.arange(zcr_feat.shape[0]))
    return zcr_feat, col_names, df


def feature_quantile_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    q: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute time-domain quantiles for each epoch × channel.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    sfreq : float, optional
        Unused, kept for consistent signature across features.
    ch_names : list[str], required
        Channel names (len == n_channels).
    q : tuple of float in [0, 1]
        Quantiles to compute (e.g., (0.25, 0.5, 0.75)).

    Returns
    -------
    q_feat_all : np.ndarray, shape (n_epochs, n_channels * len(q))
        Quantile values concatenated in channel-major order.
    col_names_all : list[str]
        Column names like 'q25_Fp1', 'q50_Fp1', 'q75_Fp1', ...
    df_all : pd.DataFrame
        DataFrame with quantile features per (channel × quantile).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    q = tuple(q)
    # Quantile calculation across time
    q_vals = np.quantile(X, q, axis=-1)  # shape (len(q), n_epochs, n_channels)
    q_vals = np.moveaxis(q_vals, 0, -1)  # → (n_epochs, n_channels, len(q))

    n_epochs, n_channels, n_q = q_vals.shape
    q_feat_all = q_vals.reshape(n_epochs, n_channels * n_q)

    # Build column names (channel-major order)
    q_perc = [int(round(100 * qi)) for qi in q]
    col_names_all = [f"q{qi}_{ch}" for ch in ch_names for qi in q_perc]

    df_all = pd.DataFrame(q_feat_all, columns=col_names_all, index=np.arange(n_epochs))
    return q_feat_all, col_names_all, df_all


def feature_hjorth_mobility_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Hjorth Mobility for each epoch × channel using mne-features.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape

    hj_mob_list = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hj_mob = mne_features.univariate.compute_hjorth_mobility(epoch_data)
        hj_mob_list.append(hj_mob)

    hj_mob_feat = np.vstack(hj_mob_list)  # (n_epochs, n_channels)
    col_names = [f"hj_mob_{ch}" for ch in ch_names]
    df = pd.DataFrame(hj_mob_feat, columns=col_names, index=np.arange(n_epochs))

    return hj_mob_feat, col_names, df


def feature_hjorth_complexity_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Hjorth Complexity for each epoch × channel using mne-features.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape

    hj_com_list = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hj_com = mne_features.univariate.compute_hjorth_complexity(epoch_data)
        hj_com_list.append(hj_com)

    hj_com_feat = np.vstack(hj_com_list)  # (n_epochs, n_channels)
    col_names = [f"hj_com_{ch}" for ch in ch_names]
    df = pd.DataFrame(hj_com_feat, columns=col_names, index=np.arange(n_epochs))

    return hj_com_feat, col_names, df

def feature_decorr_time_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Decorrelation Time for each epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    sfreq : float, required
        Sampling frequency (Hz).
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    dt_feat : np.ndarray, shape (n_epochs, n_channels)
        Decorrelation time (seconds) per epoch × channel.
    col_names : list[str]
        Column names like 'decorr_time_Fp1', 'decorr_time_Cz', ...
    df : pd.DataFrame
        DataFrame version of the results.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape

    rows = []
    for i in range(n_epochs):
        epoch_dt = mne_features.univariate.compute_decorr_time(float(sfreq), X[i]).astype(float)
        rows.append(epoch_dt)

    dt_feat = np.vstack(rows)  # (n_epochs, n_channels)

    col_names = [f"decorr_time_{ch}" for ch in ch_names]
    df = pd.DataFrame(dt_feat, columns=col_names, index=np.arange(n_epochs))
    return dt_feat, col_names, df


# -----------------------------------------------------------------------------
# FEATURE_REGISTRY_TIME
# -----------------------------------------------------------------------------
# This registry acts as a central lookup table that maps short feature names
# (used in the extract_time_domain_features() call) to their corresponding
# implementation functions.
#
# ✅ Why it’s useful:
#    - Makes it easy to dynamically select which time-domain features to extract
#      without editing the core logic.
#    - Each key corresponds to a specific EEG feature function defined above.
#    - The registry allows clean, modular control of which features are computed.
#
# Example usage:
#     X, cols, df = extract_time_domain_features(epochs, features=("mean", "rms"))
#
#     → will automatically call feature_mean_per_epoch() and feature_rms_per_epoch()
# -----------------------------------------------------------------------------

FEATURE_REGISTRY_TIME = {
    "std":        feature_std_per_epoch,              # T1 — Standard deviation of amplitude over time
    "mean":       feature_mean_per_epoch,             # T2 — Mean signal amplitude per epoch × channel
    "ptp":        feature_ptp_per_epoch,              # T3 — Peak-to-peak amplitude (max − min)
    "skew":       feature_skew_per_epoch,             # T4 — Skewness (asymmetry of amplitude distribution)
    "kurt":       feature_kurt_per_epoch,             # T5 — Kurtosis (tailedness / peakedness)
    "rms":        feature_rms_per_epoch,              # T6 — Root-mean-square (signal power measure)    
    "hj_mob":     feature_hjorth_mobility_per_epoch,  # T7 — Hjorth Mobility (ratio of signal derivative stds)
    "quantile":   feature_quantile_per_epoch,         # T8 — Quantile statistics (e.g., 25/50/75th percentiles)
    "hj_com":     feature_hjorth_complexity_per_epoch,# T9 — Hjorth Complexity (shape complexity)
    "var":        feature_var_per_epoch,              # T10 — Variance of amplitude over time
    "decorr_time":feature_decorr_time_per_epoch,      # T11 — Decorrelation time (autocorrelation decay)
    "zcr":        feature_zcr_per_epoch,              # T12 — Zero-crossing rate (sign-change count)
}

# eeg_features_time.py
FEATURE_DOCS_TIME = {
    "std":         "Standard deviation of EEG amplitude over time (per channel).",
    "mean":        "Mean EEG amplitude within each epoch (per channel).",
    "ptp":         "Peak-to-peak amplitude (max − min) per epoch per channel.",
    "skew":        "Skewness of the amplitude distribution (signal asymmetry).",
    "kurt":        "Kurtosis of amplitude distribution (tailedness/peakedness).",
    "rms":         "Root-mean-square amplitude (signal energy measure).",
    "hj_mob":      "Hjorth Mobility — ratio of std of first derivative to signal std.",
    "quantile":    "Selected quantile statistics (e.g., 25th, 50th, 75th percentiles).",
    "hj_com":      "Hjorth Complexity — shape complexity relative to a sine wave.",
    "var":         "Variance of amplitude across time (per epoch per channel).",
    "decorr_time": "Decorrelation time — lag where autocorrelation first drops below 1/e.",
    "zcr":         "Zero-crossing rate — frequency of sign changes in the signal.",
}


def extract_time_domain_features(
    epochs_or_X: Union[Any, Sequence[Any]],
    features: List[str] = ['mean', 'std'],
    sfreq: Optional[float] = None,
    ch_names: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,   # NEW: to build dfs_dict keys
    **feature_kwargs,
):
    """
    Extract time-domain EEG features.

    SINGLE INPUT (mne.Epochs or ndarray):
      returns -> (X_all, colnames_all, df_all)

    LIST INPUT (sequence of subjects):
      returns -> (features_list, colnames_all, dfs_dict)
        - features_list: [ (n_epochs_i, F_total), ... ]
        - dfs_dict: { "LABEL_SUBJECTID": per-subject DataFrame, ... }
          Requires `metadata` aligned to the list input.
    """
    # -------------------------
    # Branch: list of subjects
    # -------------------------
    if isinstance(epochs_or_X, (list, tuple)):
        subjects: Sequence[Any] = epochs_or_X
        if len(subjects) == 0:
            return [], [], {}  # empty dict when list mode

        if metadata is None or len(metadata) != len(subjects):
            raise ValueError(
                "When passing a list of subjects, provide `metadata` with the same length "
                "so we can key dfs_dict by 'LABEL_SUBJECTID'."
            )

        sf = None if sfreq is None else float(sfreq)
        chn = None if ch_names is None else list(ch_names)

        features_list: List[np.ndarray] = []
        dfs_dict: Dict[str, pd.DataFrame] = {}
        colnames_all: List[str] = []

        for i, subj in enumerate(subjects):
            # Normalize this subject to array
            Xi, sf_i, chn_i = _normalize_epochs_input(subj, sf, chn)

            
            # Compute all requested features
            arrays_i, cols_i, dfs_i = [], [], []
            for name in features:
                if name not in FEATURE_REGISTRY_TIME:
                    raise ValueError(f"Unknown time feature: '{name}'")
                fn = FEATURE_REGISTRY_TIME[name]
                kwargs = feature_kwargs.get(name, {})
                arr, cols, df = fn(Xi, sfreq=sf_i, ch_names=chn_i, **kwargs)
                arrays_i.append(arr)
                cols_i.extend(cols)
                dfs_i.append(df)

            # Concatenate results for this subject
            X_all_i = np.concatenate(arrays_i, axis=1) if len(arrays_i) > 1 else arrays_i[0]
            df_all_i = pd.concat(dfs_i, axis=1)

            # Fix canonical columns on first subject
            if i == 0:
                colnames_all = cols_i
            else:
                if cols_i != colnames_all:
                    raise ValueError(
                        f"Feature columns mismatch at subject index {i}. "
                        f"Expected {len(colnames_all)}, got {len(cols_i)}."
                    )

            # Append array
            features_list.append(X_all_i)

            # Build key "LABEL_SUBJECTID" from metadata[i]
            lab = metadata[i]["label"]
            sid = metadata[i]["subject_id"]
            key = f"{lab}_{sid}"
            dfs_dict[key] = df_all_i

        return features_list, colnames_all, dfs_dict

    # -------------------------
    # Branch: single subject
    # -------------------------
    X, sf, chn = _normalize_epochs_input(epochs_or_X, sfreq, ch_names)

    all_arrays: List[np.ndarray] = []
    all_colnames: List[str] = []
    all_dfs: List[pd.DataFrame] = []

    for name in features:
        if name not in FEATURE_REGISTRY_TIME:
            raise ValueError(f"Unknown time feature: '{name}'")
        fn = FEATURE_REGISTRY_TIME[name]
        kwargs = feature_kwargs.get(name, {})
        arr, cols, df = fn(X, sfreq=sf, ch_names=chn, **kwargs)
        all_arrays.append(arr)
        all_colnames.extend(cols)
        all_dfs.append(df)

    X_all = np.concatenate(all_arrays, axis=1) if len(all_arrays) > 1 else all_arrays[0]
    df_all = pd.concat(all_dfs, axis=1)
    return X_all, all_colnames, df_all
