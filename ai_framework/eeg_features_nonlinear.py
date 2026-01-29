# eeg_features_nonlinear.py
# Non-linear dynamic EEG feature extraction using existing functions.


from typing import List, Optional, Tuple, Dict, Any, Union, Sequence, Mapping, Literal
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import mne_features
from .utils import validate_feature_names, _normalize_epochs_input


def feature_line_length_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
):
    """
    Compute Line Length per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG array of shape (n_epochs, n_channels, n_times).
        Already normalized upstream.
    sfreq : float, optional
        Unused here—present only for API consistency.
    ch_names : list[str], required
        List of channel names (length = n_channels).

    Returns
    -------
    ll_feat : np.ndarray, shape (n_epochs, n_channels)
        Line length for each epoch and channel.
    col_names : list[str]
        Column names like 'lineLength_Fp1', 'lineLength_Cz', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Compute line length per epoch ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        ll = mne_features.univariate.compute_line_length(epoch_data)  # (n_channels,)
        rows.append(ll)

    ll_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Build column names ---
    col_names = [f"lineLength_{ch}" for ch in ch_names]

    # --- Build DataFrame ---
    df = pd.DataFrame(ll_feat, columns=col_names)

    return ll_feat, col_names, df


def feature_shannon_entropy_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
    psd_method: str = "welch",
    psd_params: dict | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Shannon (spectral) entropy per epoch × channel using mne-features.

    Shannon entropy here quantifies the irregularity or unpredictability
    of the power spectral density (PSD) of the EEG signal.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, required
        Sampling frequency (Hz).
    ch_names : list[str], required
        Channel names (len == n_channels).
    psd_method : {'welch','multitaper','fft'}, default='welch'
        PSD estimation method.
    psd_params : dict | None, default=None
        Optional parameters for PSD estimation, e.g.
        {'welch_n_fft': 512, 'welch_n_per_seg': 256, 'welch_n_overlap': 128}.

    Returns
    -------
    shan_feat : np.ndarray, shape (n_epochs, n_channels)
        Shannon entropy per epoch × channel.
    col_names : list[str]
        Column names like 'shannonEntropy_Fp1', 'shannonEntropy_Cz', ...
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if sfreq is None:
        raise ValueError("Sampling frequency (`sfreq`) must be provided.")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape
    sfreq = float(sfreq)

    # --- Per-epoch computation; mne-features expects (n_channels, n_times) ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        shan = mne_features.univariate.compute_spect_entropy(
            sfreq,
            epoch_data,
            psd_method=psd_method,
            psd_params=psd_params,
        )  # -> (n_channels,)
        rows.append(shan)

    shan_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Columns & DataFrame ---
    col_names = [f"shannonEntropy_{ch}" for ch in ch_names]
    df = pd.DataFrame(shan_feat, columns=col_names)

    return shan_feat, col_names, df

def feature_hurst_exp_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
):
    """
    Compute Hurst Exponent per epoch × channel using mne-features.

    The Hurst exponent quantifies long-term temporal dependence:
      - ~0.5 : random / white-noise-like
      - <0.5: anti-persistent
      - >0.5: persistent, long-memory behavior

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
        Already normalized upstream.
    sfreq : float, optional
        Unused here; kept for API consistency.
    ch_names : list[str], required
        Channel names (length must match n_channels).

    Returns
    -------
    hurst_feat : np.ndarray, shape (n_epochs, n_channels)
        Hurst exponent for each epoch and channel.
    col_names : list[str]
        Column names like 'hurstExp_Fp1', 'hurstExp_Cz', ...
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Per-epoch computation; mne-features expects (n_channels, n_times) ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hurst = mne_features.univariate.compute_hurst_exp(epoch_data)  # (n_channels,)
        rows.append(hurst)

    hurst_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Columns & DataFrame ---
    col_names = [f"hurstExp_{ch}" for ch in ch_names]
    df = pd.DataFrame(hurst_feat, columns=col_names)

    return hurst_feat, col_names, df

def feature_samp_entropy_per_epoch(
    X: np.ndarray,
    emb: int = 2,
    metric: str = "chebyshev",
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
):
    """
    Compute Sample Entropy (SampEn) per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data array of shape (n_epochs, n_channels, n_times).
        Already normalized upstream.
    emb : int, default=2
        Embedding dimension for SampEn.
    metric : str, default='chebyshev'
        Distance metric used internally by mne-features.
    sfreq : float, optional
        Unused here; present for API consistency with other feature functions.
    ch_names : list[str], required
        Channel names (length must match n_channels).

    Returns
    -------
    samp_feat : np.ndarray, shape (n_epochs, n_channels)
        Sample entropy per epoch and channel.
    col_names : list[str]
        Column names like 'sampEntropy_Fp1', 'sampEntropy_Cz', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Compute SampEn for each epoch ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        samp = mne_features.univariate.compute_samp_entropy(
            epoch_data, emb=emb, metric=metric
        )  # -> (n_channels,)
        rows.append(samp)

    samp_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Build column names & DataFrame ---
    col_names = [f"sampEntropy_{ch}" for ch in ch_names]
    df = pd.DataFrame(samp_feat, columns=col_names)

    return samp_feat, col_names, df

def feature_app_entropy_per_epoch(
    X: np.ndarray,
    emb: int = 2,
    metric: str = "chebyshev",
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Approximate Entropy (AppEn) per epoch × channel using mne-features.

    Approximate Entropy quantifies the regularity or predictability of a time series.
    - Lower AppEn  → more regular / predictable signal
    - Higher AppEn → more complex / irregular signal

    Parameters
    ----------
    X : np.ndarray
        EEG data array of shape (n_epochs, n_channels, n_times).
        Assumed already normalized upstream.
    emb : int, default=2
        Embedding dimension.
    metric : str, default='chebyshev'
        Distance metric used internally by mne-features / KDTree.
    sfreq : float, optional
        Unused here; present only for consistency with other feature functions.
    ch_names : list[str], required
        Channel names (length must match n_channels).

    Returns
    -------
    app_feat : np.ndarray, shape (n_epochs, n_channels)
        Approximate entropy value for each epoch and channel.
    col_names : list[str]
        Column names like 'appEntropy_Fp1', 'appEntropy_Cz', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Per-epoch computation (mne-features expects (n_channels, n_times)) ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        app = mne_features.univariate.compute_app_entropy(
            epoch_data,
            emb=emb,
            metric=metric,
        )  # -> (n_channels,)
        rows.append(app)

    app_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Column names & DataFrame ---
    col_names = [f"appEntropy_{ch}" for ch in ch_names]
    df = pd.DataFrame(app_feat, columns=col_names)

    return app_feat, col_names, df

def feature_svd_entropy_per_epoch(
    X: np.ndarray,
    tau: int = 2,
    emb: int = 10,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute SVD Entropy per epoch × channel using mne-features.

    SVD Entropy measures signal complexity based on the singular values
    of a time-delay embedded trajectory matrix. Higher values indicate
    greater complexity.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times).
    tau : int, default=2
        Delay (in samples) for time-delay embedding.
    emb : int, default=10
        Embedding dimension.
    sfreq : float, optional
        Unused here; kept for API consistency.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    svd_feat : np.ndarray, shape (n_epochs, n_channels)
        SVD entropy per epoch × channel.
    col_names : list[str]
        Column names like 'svdEntropy_Fp1', 'svdEntropy_Cz', ...
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Per-epoch computation; mne-features expects (n_channels, n_times) ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        svd = mne_features.univariate.compute_svd_entropy(
            epoch_data,
            tau=tau,
            emb=emb,
        )  # -> (n_channels,)
        rows.append(svd)

    svd_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Column names & DataFrame ---
    col_names = [f"svdEntropy_{ch}" for ch in ch_names]
    df = pd.DataFrame(svd_feat, columns=col_names)

    return svd_feat, col_names, df

def feature_higuchi_fd_per_epoch(
    X: np.ndarray,
    kmax: int = 10,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Higuchi Fractal Dimension (HFD) per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times).
        Must be preprocessed/normalized upstream.
    kmax : int, default=10
        Maximum delay/offset (in samples) used by the Higuchi algorithm.
    sfreq : float, optional
        Unused here, kept for interface consistency.
    ch_names : list[str], required
        Channel names (length must match n_channels).

    Returns
    -------
    hfd_feat : np.ndarray, shape (n_epochs, n_channels)
        Higuchi Fractal Dimension for each epoch and channel.
    col_names : list[str]
        Column names like 'higuchiFD_Fp1', 'higuchiFD_Cz', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Compute HFD per epoch ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        hfd = mne_features.univariate.compute_higuchi_fd(epoch_data, kmax=kmax)  # (n_channels,)
        rows.append(hfd)

    hfd_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Build column names & DataFrame ---
    col_names = [f"higuchiFD_{ch}" for ch in ch_names]
    df = pd.DataFrame(hfd_feat, columns=col_names)

    return hfd_feat, col_names, df

def feature_teager_kaiser_energy_per_epoch(
    X: np.ndarray,
    wavelet_name: str = "db4",
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Teager–Kaiser Energy via wavelet decomposition per epoch × channel.

    Uses mne_features.univariate.compute_teager_kaiser_energy, which for a single
    epoch (n_channels, n_times) returns a 1D vector of length:
        n_channels * (levdec + 1) * 2

    where:
      - levdec + 1 = number of wavelet scales (levels)
      - 2          = two summary values per scale

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_times). Already normalized upstream.
    wavelet_name : str, default "db4"
        Wavelet name to use (passed to mne-features / pywt).
    sfreq : float, optional
        Unused here; present for API consistency.
    ch_names : list[str], required
        Channel names (len == n_channels).

    Returns
    -------
    tke_feat : np.ndarray, shape (n_epochs, n_features)
        Teager–Kaiser energy features per epoch. n_features =
        n_channels * (levdec + 1) * 2.
    col_names : list[str]
        Column names in channel-major order, e.g.
        'teagerKaiser_L0_C1_Fp1', 'teagerKaiser_L0_C2_Fp1', ...,
        over channels, levels, and the two components.
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per feature.
    """
    import mne_features

    # --- Basic validation ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Compute TKE per epoch ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        tke = mne_features.univariate.compute_teager_kaiser_energy(
            epoch_data,
            wavelet_name=wavelet_name,
        )  # shape: (n_channels * (levdec + 1) * 2,)
        rows.append(tke)

    tke_feat = np.vstack(rows)  # (n_epochs, n_channels * (levdec + 1) * 2)

    # --- Infer number of wavelet levels from output length ---
    per_channel = tke_feat.shape[1] // n_channels
    if per_channel * n_channels != tke_feat.shape[1]:
        raise ValueError("Unexpected Teager–Kaiser feature length; cannot factor by channels cleanly.")

    # per_channel = (levdec + 1) * 2
    if per_channel % 2 != 0:
        raise ValueError("Unexpected Teager–Kaiser per-channel length; expected an even number.")
    n_levels = per_channel // 2  # number of wavelet scales

    # --- Build column names: channel-major → for each channel, level, component ---
    col_names = [
        f"teagerKaiser_L{L}_C{c}_{ch}"
        for ch in ch_names
        for L in range(n_levels)
        for c in (1, 2)  # the two summary components per level
    ]

    df = pd.DataFrame(tke_feat, columns=col_names)
    return tke_feat, col_names, df

def feature_svd_fisher_info_per_epoch(
    X: np.ndarray,
    tau: int = 2,
    emb: int = 10,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute SVD Fisher Information per epoch × channel using mne-features.

    Parameters
    ----------
    X : np.ndarray
        EEG array of shape (n_epochs, n_channels, n_times).
        Already normalized upstream.
    tau : int, default=2
        Delay (number of samples) for time-delay embedding.
    emb : int, default=10
        Embedding dimension.
    sfreq : float, optional
        Unused here; included for API consistency.
    ch_names : list[str], required
        Channel names (length must equal n_channels).

    Returns
    -------
    svd_fi_feat : np.ndarray, shape (n_epochs, n_channels)
        SVD Fisher Information per epoch and channel.
    col_names : list[str]
        Column names like 'svdFisherInfo_Fp1', 'svdFisherInfo_Cz', ...
    df : pd.DataFrame
        One row per epoch, one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array of shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Per-epoch computation; mne-features expects (n_channels, n_times) ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        svd_fi = mne_features.univariate.compute_svd_fisher_info(
            epoch_data,
            tau=tau,
            emb=emb,
        )  # (n_channels,)
        rows.append(svd_fi)

    svd_fi_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Column names & DataFrame ---
    col_names = [f"svdFisherInfo_{ch}" for ch in ch_names]
    df = pd.DataFrame(svd_fi_feat, columns=col_names)

    return svd_fi_feat, col_names, df


def feature_katz_fd_per_epoch(
    X: np.ndarray,
    sfreq: float | None = None,
    ch_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Compute Katz Fractal Dimension (KFD) per epoch × channel using mne-features.

    The Katz Fractal Dimension quantifies the complexity of a signal based on the
    ratio between its total length and the maximum distance between any two points.
    Higher values correspond to higher complexity.

    Parameters
    ----------
    X : np.ndarray
        EEG data of shape (n_epochs, n_channels, n_times). Already normalized upstream.
    sfreq : float, optional
        Unused here; present for interface consistency across features.
    ch_names : list[str], required
        Channel names (length must equal n_channels).

    Returns
    -------
    kfd_feat : np.ndarray, shape (n_epochs, n_channels)
        Katz Fractal Dimension per epoch and channel.
    col_names : list[str]
        Column names like 'katzFD_Fp1', 'katzFD_Cz', ...
    df : pd.DataFrame
        DataFrame with one row per epoch and one column per channel.
    """
    # --- Validate inputs ---
    if X.ndim != 3:
        raise ValueError(f"Expected array shape (n_epochs, n_channels, n_times); got {X.shape}")
    if ch_names is None or len(ch_names) != X.shape[1]:
        raise ValueError("ch_names must be provided and match X.shape[1].")

    n_epochs, n_channels, _ = X.shape

    # --- Per-epoch KFD computation ---
    rows = []
    for i in range(n_epochs):
        epoch_data = X[i]  # (n_channels, n_times)
        kfd = mne_features.univariate.compute_katz_fd(epoch_data)  # (n_channels,)
        rows.append(kfd)

    kfd_feat = np.vstack(rows)  # (n_epochs, n_channels)

    # --- Columns & DataFrame ---
    col_names = [f"katzFD_{ch}" for ch in ch_names]
    df = pd.DataFrame(kfd_feat, columns=col_names)

    return kfd_feat, col_names, df


# -----------------------------------------------------------------------------
# FEATURE_REGISTRY_NLD
# -----------------------------------------------------------------------------
FEATURE_REGISTRY_NLD = {
    "lineLength":     feature_line_length_per_epoch,        # NL1 — Line length (signal roughness / total variation)
    "shannonEntropy": feature_shannon_entropy_per_epoch,    # NL2 — Shannon entropy on PSD (spectral irregularity)
    "hurstExp":       feature_hurst_exp_per_epoch,          # NL3 — Hurst exponent (long-range dependence)
    "sampEntropy":    feature_samp_entropy_per_epoch,       # NL4 — Sample entropy (regularity/complexity)
    "appEntropy":     feature_app_entropy_per_epoch,        # NL5 — Approximate entropy (predictability)
    "svdEntropy":     feature_svd_entropy_per_epoch,        # NL6 — SVD entropy (embedding-based complexity)
    "higuchiFD":      feature_higuchi_fd_per_epoch,         # NL7 — Higuchi fractal dimension (self-similarity)
    "teagerKaiser":   feature_teager_kaiser_energy_per_epoch,# NL8 — Teager–Kaiser energy via wavelets (per scale)
    "svdFisherInfo":  feature_svd_fisher_info_per_epoch,    # NL9 — SVD Fisher information (structure in embedding)
    "katzFD":         feature_katz_fd_per_epoch,            # NL10 — Katz fractal dimension (curve complexity)
}

FEATURE_DOCS_NLD = {
    "lineLength": "Signal roughness / total variation per channel.",
    "shannonEntropy": "Spectral entropy from PSD (irregularity).",
    "hurstExp": "Long-range dependence (Hurst exponent).",
    "sampEntropy": "Sample entropy (regularity/complexity).",
    "appEntropy": "Approximate entropy (predictability).",
    "svdEntropy": "SVD entropy via time-delay embedding.",
    "higuchiFD": "Higuchi fractal dimension (self-similarity).",
    "teagerKaiser": "Teager–Kaiser energy via wavelets (per scale).",
    "svdFisherInfo": "SVD Fisher information (structure in embedding).",
    "katzFD": "Katz fractal dimension (curve complexity).",
}



def extract_nonlineardynamics_features(
    epochs_or_X: Union[Any, Sequence[Any]],
    features: List[str] = [
        "lineLength",
        "shannonEntropy",
        "hurstExp",
        "sampEntropy",
        "appEntropy",
        "svdEntropy",
        "higuchiFD",
        "teagerKaiser",
        "svdFisherInfo",
        "katzFD",
    ],
    sfreq: Optional[float] = None,
    ch_names: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,   # for dict keys in list mode
    **feature_kwargs,
):
    """
    Extract nonlinear dynamics EEG features.

    SINGLE INPUT (mne.Epochs or ndarray):
      returns -> (X_all, colnames_all, df_all)

    LIST INPUT (sequence of subjects):
      returns -> (features_list, colnames_all, dfs_dict)
        - features_list: [ (n_epochs_i, F_total), ... ]
        - dfs_dict: { "LABEL_SUBJECTID": per-subject DataFrame, ... }
          `metadata` must be aligned 1:1 with the list input.

    Parameters
    ----------
    epochs_or_X : mne.Epochs | np.ndarray | sequence of those
        Either a single subject (MNE Epochs or 3D ndarray) or a list of subjects.
    features : list[str]
        Names registered in FEATURE_REGISTRY_NLD, e.g.
        ["lineLength", "shannonEntropy", "hurstExp", ...].
    sfreq : float, optional
        Sampling frequency in Hz. Required when passing raw ndarrays.
        Ignored when epochs_or_X is an mne.Epochs object.
    ch_names : list[str], optional
        Channel names for ndarray input. Ignored for mne.Epochs.
    metadata : list[dict], optional
        Per-subject metadata from `build_label_epoch_arrays`. Must contain "label"
        and "subject_id" for each entry if you want a dfs_dict keyed by "LABEL_SUBJECTID".
    **feature_kwargs : dict
        Per-feature keyword arguments, e.g.
        {
            "sampEntropy": {"m": 2, "r": 0.2},
            "appEntropy": {"m": 2, "r": 0.2},
        }
    """
    # -------------------------
    # Branch: list of subjects
    # -------------------------
    if isinstance(epochs_or_X, (list, tuple)):
        subjects: Sequence[Any] = epochs_or_X
        if len(subjects) == 0:
            return [], [], {}

        # We need metadata to build per-subject keys in dfs_dict
        if metadata is None or len(metadata) != len(subjects):
            raise ValueError(
                "When passing a list of subjects, provide `metadata` with the same length "
                "so we can key dfs_dict by 'LABEL_SUBJECTID'."
            )

        # Optional global sfreq / ch_names; can be overridden by each subject if it's MNE Epochs
        sf = None if sfreq is None else float(sfreq)
        chn = None if ch_names is None else list(ch_names)

        features_list: List[np.ndarray] = []
        dfs_dict: Dict[str, pd.DataFrame] = {}
        colnames_all: List[str] = []

        # Loop each subject in order
        for i, subj in enumerate(subjects):
            # 1) Normalize this subject to array (n_epochs, n_channels, n_times)
            Xi, sf_i, chn_i = _normalize_epochs_input(subj, sf, chn)

            # 2) Compute all requested nonlinear features for this subject
            arrays_i: List[np.ndarray] = []
            cols_i: List[str] = []
            dfs_i: List[pd.DataFrame] = []

            for name in features:
                if name not in FEATURE_REGISTRY_NLD:
                    raise ValueError(f"Unknown nonlinear feature: '{name}'")

                fn = FEATURE_REGISTRY_NLD[name]
                kwargs = feature_kwargs.get(name, {})

                # All NLD feature fns are expected to accept (X, sfreq, ch_names, **kwargs)
                arr, cols, df = fn(Xi, sfreq=sf_i, ch_names=chn_i, **kwargs)

                arrays_i.append(arr)
                cols_i.extend(cols)
                dfs_i.append(df)

            # 3) Concatenate features for this subject
            X_all_i = np.concatenate(arrays_i, axis=1) if len(arrays_i) > 1 else arrays_i[0]
            df_all_i = pd.concat(dfs_i, axis=1)

            # 4) Capture canonical column names from the first subject
            if i == 0:
                colnames_all = cols_i
            else:
                if cols_i != colnames_all:
                    raise ValueError(
                        f"Nonlinear feature columns mismatch at subject index {i}. "
                        f"Expected {len(colnames_all)} columns, got {len(cols_i)}."
                    )

            # 5) Append this subject's feature matrix
            features_list.append(X_all_i)

            # 6) Build per-subject key: "LABEL_SUBJECTID"
            lab = metadata[i]["label"]
            sid = metadata[i]["subject_id"]
            key = f"{lab}_{sid}"
            dfs_dict[key] = df_all_i

        return features_list, colnames_all, dfs_dict

    # -------------------------
    # Branch: single subject
    # -------------------------
    # Normalize to (n_epochs, n_channels, n_times)
    X, sf, chn = _normalize_epochs_input(epochs_or_X, sfreq, ch_names)

    arrays: List[np.ndarray] = []
    names: List[str] = []
    dfs: List[pd.DataFrame] = []

    # Loop over requested nonlinear features
    for name in features:
        if name not in FEATURE_REGISTRY_NLD:
            raise ValueError(f"Unknown nonlinear feature: '{name}'")

        fn = FEATURE_REGISTRY_NLD[name]
        kwargs = feature_kwargs.get(name, {})

        arr, cols, df = fn(X, sfreq=sf, ch_names=chn, **kwargs)
        arrays.append(arr)
        names.extend(cols)
        dfs.append(df)

    # Concatenate feature blocks for this single subject
    X_all = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]
    df_all = pd.concat(dfs, axis=1)

    return X_all, names, df_all

    


def extract_nonlineardynamics_features_parallel(
    epochs_or_X: Union[Any, Sequence[Any]],
    features: List[str] = [
        "lineLength",
        "shannonEntropy",
        "hurstExp",
        "sampEntropy",
        "appEntropy",
        "svdEntropy",
        "higuchiFD",
        "teagerKaiser",
        "svdFisherInfo",
        "katzFD",
    ],
    sfreq: Optional[float] = None,
    ch_names: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,   # required for list-mode keying
    *,
    n_jobs: int = -1,
    backend: Literal["threading", "loky"] = "loky",
    verbose: bool = False,
    **feature_kwargs: Any,
) -> Union[
    Tuple[np.ndarray, List[str], pd.DataFrame],
    Tuple[List[np.ndarray], List[str], Dict[str, pd.DataFrame]],
]:
    """
    Extract nonlinear dynamics EEG features with **per-feature parallelization**.

    SINGLE INPUT (mne.Epochs or ndarray):
      returns -> (X_all, colnames_all, df_all)

    LIST INPUT (sequence of subjects):
      returns -> (features_list, colnames_all, dfs_dict)
        - features_list: [ (n_epochs_i, F_total), ... ]
        - dfs_dict: { "LABEL_SUBJECTID": per-subject DataFrame, ... }
          `metadata` must be aligned 1:1 with the list input.

    Parallelization (simple):
      - For EACH subject: run each feature in `features` in parallel via joblib.
      - Then concatenate feature blocks in the ORIGINAL `features` order.

    Parameters
    ----------
    epochs_or_X : mne.Epochs | np.ndarray | sequence of those
    features : list[str]
        Must be keys in FEATURE_REGISTRY_NLD.
    sfreq : float, optional
        Needed for ndarray inputs when required by a feature (e.g., shannonEntropy).
    ch_names : list[str], optional
        Required for ndarray inputs (ignored for mne.Epochs input).
    metadata : list[dict], optional
        Required in list-mode. Each dict must have 'label' and 'subject_id'.
    n_jobs : int
        Joblib workers for per-feature parallelism. Use -1 for all cores; 1 disables.
    backend : {"threading","loky"}
    verbose : bool
        If True, prints subject index and shape in list-mode.
    **feature_kwargs : dict
        Per-feature kwargs mapping. Example call:
          extract_nonlineardynamics_features_parallel(
              ...,
              sampEntropy={"emb": 2, "metric": "chebyshev"},
              svdEntropy={"tau": 2, "emb": 10},
          )

    Raises
    ------
    ValueError:
        - empty features
        - unknown feature name
        - list-mode metadata missing/mismatched
        - feature column mismatch across subjects
        - feature_kwargs for a feature is not dict-like
    """
    from joblib import Parallel, delayed

    # -------------------------
    # Basic validation
    # -------------------------
    if not features:
        raise ValueError("`features` must be a non-empty list of feature names.")

    unknown = [f for f in features if f not in FEATURE_REGISTRY_NLD]
    if unknown:
        raise ValueError(f"Unknown nonlinear feature(s): {unknown}")

    # Per-feature kwargs are passed as **feature_kwargs; interpret as name -> dict
    per_feature_params: Dict[str, Any] = dict(feature_kwargs)

    # -------------------------
    # One feature computation
    # -------------------------
    def _run_feature(
        name: str,
        X_in: np.ndarray,
        sf_in: float,
        ch_in: List[str],
    ) -> Tuple[str, np.ndarray, List[str], pd.DataFrame]:
        """Compute one feature block (arr, cols, df) for one subject."""
        fn = FEATURE_REGISTRY_NLD[name]
        kwargs_for_feature = per_feature_params.get(name, {})
        if kwargs_for_feature is None:
            kwargs_for_feature = {}
        if not isinstance(kwargs_for_feature, Mapping):
            raise ValueError(
                f"feature_kwargs for '{name}' must be a dict-like mapping, got {type(kwargs_for_feature)}"
            )
        arr, cols, df = fn(X_in, sfreq=sf_in, ch_names=ch_in, **kwargs_for_feature)
        return name, arr, cols, df

    # -------------------------
    # Compute all features for one subject (parallel across FEATURES)
    # -------------------------
    def _compute_subject(
        subj_obj: Any,
        sf_global: Optional[float],
        ch_global: Optional[List[str]],
    ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        # Normalize to (n_epochs, n_channels, n_times), plus sfreq + channel names
        X, sf, chn = _normalize_epochs_input(subj_obj, sf_global, ch_global)

        use_parallel = (n_jobs != 1) and (len(features) > 1)

        if use_parallel:
            # Disable memmapping when using loky so X doesn't become a read-only memmap
            parallel_kwargs = {}
            if backend == "loky":
                parallel_kwargs["max_nbytes"] = None
                parallel_kwargs["mmap_mode"] = None

            results = Parallel(n_jobs=n_jobs, backend=backend, **parallel_kwargs)(
                delayed(_run_feature)(name, X, sf, chn) for name in features
            )
        else:
            results = [_run_feature(name, X, sf, chn) for name in features]

        # Reassemble in requested order (joblib may complete out-of-order)
        # results items: (name, arr, cols, df)
        res_map = {name: (arr, cols, df) for (name, arr, cols, df) in results}

        arrays: List[np.ndarray] = []
        cols_all: List[str] = []
        dfs: List[pd.DataFrame] = []

        for name in features:
            arr, cols, df = res_map[name]
            arrays.append(arr)
            cols_all.extend(cols)
            dfs.append(df)

        X_all = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]
        df_all = pd.concat(dfs, axis=1)
        return X_all, cols_all, df_all


    # =====================================================================
    # LIST MODE: subjects list/tuple
    # =====================================================================
    if isinstance(epochs_or_X, (list, tuple)):
        subjects = epochs_or_X
        if len(subjects) == 0:
            return [], [], {}

        if metadata is None or len(metadata) != len(subjects):
            raise ValueError(
                "When passing a list of subjects, provide `metadata` with the same length "
                "so we can key dfs_dict by 'LABEL_SUBJECTID'."
            )

        sf_global = None if sfreq is None else float(sfreq)
        ch_global = None if ch_names is None else list(ch_names)

        features_list: List[np.ndarray] = []
        dfs_dict: Dict[str, pd.DataFrame] = {}
        colnames_all: List[str] = []

        for i, subj in enumerate(subjects):
            if verbose:
                print(f"[NLD] Subject {i}: shape={getattr(subj, 'shape', 'unknown')}")

            X_all_i, cols_i, df_all_i = _compute_subject(subj, sf_global, ch_global)

            # Enforce identical columns across subjects (first subject defines canon)
            if i == 0:
                colnames_all = cols_i
            elif cols_i != colnames_all:
                raise ValueError(
                    f"Nonlinear feature columns mismatch at subject index {i}. "
                    f"Expected {len(colnames_all)} columns, got {len(cols_i)}."
                )

            features_list.append(X_all_i)

            meta_i = metadata[i]
            if "label" not in meta_i or "subject_id" not in meta_i:
                raise ValueError(
                    f"metadata[{i}] must contain keys 'label' and 'subject_id'. "
                    f"Got keys: {list(meta_i.keys())}"
                )
            key = f"{meta_i['label']}_{meta_i['subject_id']}"
            dfs_dict[key] = df_all_i

        return features_list, colnames_all, dfs_dict

    # =====================================================================
    # SINGLE MODE: one subject
    # =====================================================================
    X_all, cols_all, df_all = _compute_subject(epochs_or_X, sfreq, ch_names)
    return X_all, cols_all, df_all