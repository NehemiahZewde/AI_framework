# eeg_features_frequency.py
# Frequency-domain EEG feature extraction using existing functions.


from typing import List, Optional, Tuple, Dict, Any, Union, Sequence
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import mne_features
from .utils import validate_feature_names, _normalize_epochs_input

def feature_wavelet_energy_per_epoch(
    X: np.ndarray,
    wavelet_name: str = "db4",
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute wavelet coefficient energy for each epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times). Already normalized upstream.
    wavelet_name : str
        PyWavelets wavelet name (e.g., 'db4').
    sfreq : float, optional
        Unused; kept for signature consistency across features.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    wavE_feat : np.ndarray, shape (n_epochs, n_channels * n_levels)
        Concatenated wavelet energy features per epoch.
    col_names : list[str]
        Column names like 'wavE_L1_Fp1', 'wavE_L2_Fp1', ..., in channel-major order.
    df : pd.DataFrame
        DataFrame with columns = col_names.
    """

    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape

    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        wavE = mne_features.univariate.compute_wavelet_coef_energy(
            epoch_data, wavelet_name=wavelet_name
        )  # shape: (n_channels * n_levels,)
        rows.append(wavE)

    wavE_feat = np.vstack(rows)  # (n_epochs, n_channels * n_levels)

    # Infer number of levels from feature length
    if wavE_feat.shape[1] % n_channels != 0:
        raise ValueError("Wavelet energy output does not align with channel count.")
    n_levels = wavE_feat.shape[1] // n_channels

    # Column names: channel-major order → [ch1_L1, ch1_L2, ..., ch2_L1, ...]
    col_names = [f"wavE_L{L}_{ch}" for ch in ch_names for L in range(1, n_levels + 1)]

    df = pd.DataFrame(wavE_feat, columns=col_names, index=np.arange(n_epochs))
    return wavE_feat, col_names, df

def feature_hjorth_complexity_spect_per_epoch(
    X: np.ndarray,
    normalize: bool = False,
    psd_method: str = "welch",
    psd_params: dict | None = None,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute spectral Hjorth Complexity per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    normalize : bool, default False
        If True, normalize result by total power.
    psd_method : {'welch', 'multitaper', 'fft'}, default 'welch'
        PSD estimation method.
    psd_params : dict | None
        Extra params for PSD (e.g., {'welch_n_fft': 512, 'welch_n_per_seg': 256}).
    sfreq : float, required
        Sampling frequency in Hz.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    hjc_feat : np.ndarray, shape (n_epochs, n_channels)
        Spectral Hjorth complexity per epoch × channel.
    col_names : list[str]
        Column names like 'hj_com_spect_Fp1', ...
    df : pd.DataFrame
        DataFrame with columns = col_names.
    """
    import mne_features

    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape

    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hjc = mne_features.univariate.compute_hjorth_complexity_spect(
            float(sfreq),
            epoch_data,
            normalize=normalize,
            psd_method=psd_method,
            psd_params=psd_params,
        )  # -> (n_channels,)
        rows.append(hjc)

    hjc_feat = np.vstack(rows)  # (n_epochs, n_channels)
    col_names = [f"hj_com_spect_{ch}" for ch in ch_names]
    df = pd.DataFrame(hjc_feat, columns=col_names, index=np.arange(n_epochs))
    return hjc_feat, col_names, df

def feature_pow_freq_bands_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    freq_bands: np.ndarray = np.array([0.5, 4., 8., 13., 30., 100.]),
    normalize: bool = True,
    ratios: str | None = None,
    ratios_triu: bool = False,
    psd_method: str = "welch",
    log: bool = False,
    psd_params: dict | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute band power per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    sfreq : float, required
        Sampling frequency (Hz).
    freq_bands : np.ndarray
        Band edges (Hz), e.g. [0.5, 4, 8, 13, 30, 100] -> 5 bands.
    normalize : bool
        Normalize band powers by total power (mne-features behavior).
    ratios : str | None
        Optional ratio mode passed to mne-features (e.g., 'pairwise', 'all').
    ratios_triu : bool
        If True and ratios set, keep only upper-triangular ratios.
    psd_method : str
        PSD method for mne-features ('welch', ...).
    log : bool
        If True, apply log to PSD before integration (mne-features flag).
    psd_params : dict | None
        Extra params for the PSD method.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    pow_feat : np.ndarray, shape (n_epochs, n_channels * n_bands)
        Band power features stacked channel-major.
    col_names : list[str]
        Column names like 'powB_B1_Fp1', ..., 'powB_Bn_Cz'.
    df : pd.DataFrame
        DataFrame with columns = col_names.
    """
    import mne_features

    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape
    sf = float(sfreq)

    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        feat = mne_features.univariate.compute_pow_freq_bands(
            sf,
            epoch_data,
            freq_bands=freq_bands,
            normalize=normalize,
            ratios=ratios,
            ratios_triu=ratios_triu,
            psd_method=psd_method,
            log=log,
            psd_params=psd_params,
        )  # shape: (n_channels * n_bands,)  (or + ratios dims if used)
        rows.append(feat)

    pow_feat = np.vstack(rows)  # (n_epochs, n_channels * n_bands[+ratios])

    # Infer number of “bands-like” slots per channel from output width
    if pow_feat.shape[1] % n_channels != 0:
        raise ValueError("Band-power output width is not divisible by n_channels. "
                         "Check freq_bands/ratios settings.")
    n_slots = pow_feat.shape[1] // n_channels

    # Column names (channel-major): [ch1_B1..B{n_slots}, ch2_B1.., ...]
    col_names = [f"powB_B{b}_{ch}" for ch in ch_names for b in range(1, n_slots + 1)]

    df = pd.DataFrame(pow_feat, columns=col_names, index=np.arange(n_epochs))
    return pow_feat, col_names, df

def feature_spect_slope_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    fmin: float = 0.1,
    fmax: float = 50.0,
    with_intercept: bool = True,
    psd_method: str = "welch",
    psd_params: dict | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Linear regression of log–log PSD (per channel) for each epoch using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    sfreq : float, required
        Sampling frequency (Hz).
    fmin, fmax : float
        Frequency range for PSD fit.
    with_intercept : bool
        If True, return intercept along with slope, mse, r2.
    psd_method : {'welch', ...}
        PSD method forwarded to mne-features.
    psd_params : dict or None
        Extra PSD kwargs (e.g., n_per_seg).
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    spec_feat : np.ndarray, shape (n_epochs, n_channels * n_metrics)
        Flattened per-epoch features in channel-major order.
    col_names : list[str]
        Columns like 'specSlope_intercept_Fp1', 'specSlope_slope_Fp1', ...
    df : pd.DataFrame
        DataFrame with columns = col_names.
    """
    import mne_features

    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        feat = mne_features.univariate.compute_spect_slope(
            float(sfreq),
            epoch_data,
            fmin=fmin,
            fmax=fmax,
            with_intercept=with_intercept,
            psd_method=psd_method,
            psd_params=psd_params,
        )
        rows.append(feat)

    spec_feat = np.vstack(rows)  # (n_epochs, n_channels * n_metrics)

    metric_names = ["intercept", "slope", "mse", "r2"] if with_intercept else ["slope", "mse", "r2"]
    col_names = [f"specSlope_{m}_{ch}" for ch in ch_names for m in metric_names]

    df = pd.DataFrame(spec_feat, columns=col_names, index=np.arange(n_epochs))
    return spec_feat, col_names, df

def feature_hjorth_mobility_spect_per_epoch(
    X: np.ndarray,
    normalize: bool = False,
    psd_method: str = "welch",
    psd_params: dict | None = None,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Spectral Hjorth Mobility per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times). Already normalized upstream.
    normalize : bool, default=False
        Whether to normalize the PSD before computing mobility.
    psd_method : str, default="welch"
        PSD estimation method to use (passed to mne-features).
    psd_params : dict | None
        Optional parameters for the PSD computation.
    sfreq : float, required
        Sampling frequency in Hz.
    ch_names : list[str], required
        List of channel names, length must equal n_channels.

    Returns
    -------
    hjm_feat : np.ndarray, shape (n_epochs, n_channels)
        Spectral Hjorth Mobility for each epoch × channel.
    col_names : list[str]
        Column names like 'hj_mob_spect_Fp1', 'hj_mob_spect_Fz', etc.
    df : pd.DataFrame
        DataFrame version of hjm_feat with labeled columns.
    """

    # --- Validate input array and parameters ---
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide `ch_names` with length matching n_channels.")

    # --- Extract shape info ---
    n_epochs, n_channels, _ = X.shape

    # --- Initialize container for results ---
    hjm_feat = np.zeros((n_epochs, n_channels), dtype=float)

    # --- Loop over epochs and compute spectral Hjorth mobility ---
    for i in range(n_epochs):
        epoch_data = X[i]  # shape: (n_channels, n_times)
        hjm_feat[i, :] = mne_features.univariate.compute_hjorth_mobility_spect(
            float(sfreq),
            epoch_data,
            normalize=normalize,
            psd_method=psd_method,
            psd_params=psd_params,
        )

    # --- Build column names ---
    col_names = [f"hj_mob_spect_{ch}" for ch in ch_names]

    # --- Convert to DataFrame for convenience ---
    df = pd.DataFrame(hjm_feat, columns=col_names, index=np.arange(n_epochs))

    return hjm_feat, col_names, df

def feature_spect_edge_freq_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ref_freq: float | None = None,
    edge: list[float] | None = None,
    psd_method: str = "welch",
    psd_params: dict | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Spectral Edge Frequency (per channel, per epoch) using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, required
        Sampling frequency in Hz.
    ref_freq : float, optional
        Reference maximum frequency for the PSD (passed to mne-features).
    edge : list[float], optional
        List of cumulative power percentages (0–100) at which to compute edge frequencies.
        Example: [95.0] for 95% spectral edge.
    psd_method : str
        PSD estimation method (e.g., "welch").
    psd_params : dict, optional
        Additional parameters passed to the PSD function.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    sef_feat : np.ndarray, shape (n_epochs, n_channels * n_edge)
        Edge frequencies concatenated in channel-major order:
        [ch1_edge1, ch1_edge2, ..., ch2_edge1, ch2_edge2, ...].
    col_names : list[str]
        Column names like 'specEdge_E1_Fp1', 'specEdge_E2_Fp1', ...
    df : pd.DataFrame
        DataFrame with columns = col_names and rows = epochs.
    """
    import mne_features

    # ----------------------------------------------------------------------
    # 1) Sanity checks on shape and metadata
    # ----------------------------------------------------------------------
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    n_epochs, n_channels, _ = X.shape

    # ----------------------------------------------------------------------
    # 2) Compute spectral edge frequency per epoch
    #    mne-features expects (n_channels, n_times) → (n_channels * n_edge,)
    # ----------------------------------------------------------------------
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        feat = mne_features.univariate.compute_spect_edge_freq(
            float(sfreq),
            epoch_data,
            ref_freq=ref_freq,
            edge=edge,
            psd_method=psd_method,
            psd_params=psd_params,
        )
        rows.append(feat)

    sef_feat = np.vstack(rows)  # (n_epochs, n_channels * n_edge)

    # ----------------------------------------------------------------------
    # 3) Infer number of edge values per channel and build column names
    # ----------------------------------------------------------------------
    if sef_feat.shape[1] % n_channels != 0:
        raise ValueError("Spectral edge output does not align with channel count.")
    n_edge = sef_feat.shape[1] // n_channels

    # channel-major: all edges for ch1, then ch2, etc.
    col_names = [
        f"specEdge_E{e}_{ch}"
        for ch in ch_names
        for e in range(1, n_edge + 1)
    ]

    # ----------------------------------------------------------------------
    # 4) Convert to DataFrame for convenience
    # ----------------------------------------------------------------------
    df = pd.DataFrame(sef_feat, columns=col_names, index=np.arange(n_epochs))
    return sef_feat, col_names, df


def feature_energy_freq_bands_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    freq_bands = np.array([0.5, 4., 8., 13., 30., 100.]),
    deriv_filt: bool = False,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Band energy (per channel) for each epoch using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, required
        Sampling frequency in Hz.
    freq_bands : array-like
        Monotonic sequence of band edges in Hz, e.g. [0.5, 4, 8, 13, 30, 100].
    deriv_filt : bool
        Whether to apply derivative filtering (passed to mne-features).
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    eng_feat : np.ndarray, shape (n_epochs, n_channels * n_bands)
        Band energy per epoch × (channel × band).
    col_names : list[str]
        Column names like 'bandEnergy_B1_Fp1', 'bandEnergy_B2_Fp1', ...
    df : pd.DataFrame
        DataFrame with columns = col_names and rows = epochs.
    """
    import mne_features

    # (n_epochs, n_channels, n_times)
    if X.ndim != 3:
        raise ValueError(f"Expected ndarray (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("Provide ch_names with length matching n_channels.")

    sfreq = float(sfreq)
    n_epochs, n_channels, _ = X.shape

    # one call per epoch; expects (n_channels, n_times)
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]
        feat = mne_features.univariate.compute_energy_freq_bands(
            sfreq,
            epoch_data,
            freq_bands=freq_bands,
            deriv_filt=deriv_filt,
        )
        rows.append(feat)

    eng_feat = np.vstack(rows)  # (n_epochs, n_channels * n_bands)

    # infer number of bands and build names
    if eng_feat.shape[1] % n_channels != 0:
        raise ValueError("Band energy output does not align with channel count.")
    n_bands = eng_feat.shape[1] // n_channels

    col_names = [
        f"bandEnergy_B{b}_{ch}"
        for ch in ch_names
        for b in range(1, n_bands + 1)
    ]

    df = pd.DataFrame(eng_feat, columns=col_names, index=np.arange(n_epochs))
    return eng_feat, col_names, df


# -----------------------------------------------------------------------------
# FEATURE_REGISTRY_FREQ
# -----------------------------------------------------------------------------
FEATURE_REGISTRY_FREQ = {
    "wavE":            feature_wavelet_energy_per_epoch,          # F1 — Wavelet Coefficient Energy
    "hj_com_spect":    feature_hjorth_complexity_spect_per_epoch, # F2 — Spectral Hjorth Complexity
    "powB":            feature_pow_freq_bands_per_epoch,          # F3 — Power in standard frequency bands (δ, θ, α, β, γ)
    "specSlope":       feature_spect_slope_per_epoch,             # F4 — Spectral slope & intercept (1/f characteristics)
    "hj_mob_spect":    feature_hjorth_mobility_spect_per_epoch,   # F2b — Spectral Hjorth Mobility
    "specEdge":        feature_spect_edge_freq_per_epoch,         # F5 — Spectral Edge Frequency (e.g., 95% power cutoff)
    "bandEnergy":      feature_energy_freq_bands_per_epoch,       # F6 — Band energy (frequency-band–specific total energy)
}

FEATURE_DOCS_FREQ = {
    "wavE": "Energy of wavelet decomposition coefficients.",
    "hj_com_spect": "Hjorth complexity computed from the power spectrum.",
    "powB": "Band power in named frequency bands (e.g., delta…gamma).",
    "specSlope": "Linear regression slope of log–log power spectrum (1/f).",
    "hj_mob_spect": "Hjorth mobility computed from the power spectrum.",
    "specEdge": "Spectral edge frequency (e.g., 95% cumulative power).",
    "bandEnergy": "Energy in defined frequency bands.",
}



def extract_frequency_domain_features(
    epochs_or_X: Union[Any, Sequence[Any]],
    features: List[str] = ["wavE", "hj_com_spect", "powB", "specSlope", "hj_mob_spect", "specEdge", "bandEnergy"],
    sfreq: Optional[float] = None,
    ch_names: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,   # for dict keys in list mode
    **feature_kwargs,
):
    """
    Extract frequency-domain EEG features.

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
            return [], [], {}

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

            # Compute all requested features for this subject
            arrays_i, cols_i, dfs_i = [], [], []
            for name in features:
                if name not in FEATURE_REGISTRY_FREQ:
                    raise ValueError(f"Unknown frequency feature: '{name}'")
                fn = FEATURE_REGISTRY_FREQ[name]
                kwargs = feature_kwargs.get(name, {})
                arr, cols, df = fn(Xi, sfreq=sf_i, ch_names=chn_i, **kwargs)
                arrays_i.append(arr)
                cols_i.extend(cols)
                dfs_i.append(df)

            # Concatenate per-subject results
            X_all_i = np.concatenate(arrays_i, axis=1) if len(arrays_i) > 1 else arrays_i[0]
            df_all_i = pd.concat(dfs_i, axis=1)

            # Fix canonical column order from first subject
            if i == 0:
                colnames_all = cols_i
            else:
                if cols_i != colnames_all:
                    raise ValueError(
                        f"Frequency feature columns mismatch at subject index {i}. "
                        f"Expected {len(colnames_all)}, got {len(cols_i)}."
                    )

            features_list.append(X_all_i)

            # Key: "LABEL_SUBJECTID"
            lab = metadata[i]["label"]
            sid = metadata[i]["subject_id"]
            key = f"{lab}_{sid}"
            dfs_dict[key] = df_all_i

        return features_list, colnames_all, dfs_dict

    # -------------------------
    # Branch: single subject
    # -------------------------
    X, sf, chn = _normalize_epochs_input(epochs_or_X, sfreq, ch_names)

    arrays: List[np.ndarray] = []
    names: List[str] = []
    dfs: List[pd.DataFrame] = []

    for name in features:
        if name not in FEATURE_REGISTRY_FREQ:
            raise ValueError(f"Unknown frequency feature: '{name}'")
        fn = FEATURE_REGISTRY_FREQ[name]
        kwargs = feature_kwargs.get(name, {})
        arr, cols, df = fn(X, sfreq=sf, ch_names=chn, **kwargs)
        arrays.append(arr)
        names.extend(cols)
        dfs.append(df)

    X_all = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]
    df_all = pd.concat(dfs, axis=1)
    return X_all, names, df_all



