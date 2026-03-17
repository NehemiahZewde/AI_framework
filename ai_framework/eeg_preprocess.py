# eeg_preprocess.py

from __future__ import annotations

import matplotlib.pyplot as plt
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import mne
from autoreject import Ransac
from mne.preprocessing import compute_current_source_density
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
import pandas as pd
from scipy.io import loadmat
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from mne_icalabel.config import ICALABEL_METHODS_NUMERICAL_TO_STRING

State = Dict[str, Any]
Params = Dict[str, Any]
StepFn = Callable[[State, Params, bool], State]


# ----------------------------
# Step functions (your style)
# ----------------------------

def step_load_eeg(state: State, params: Params, verbose: bool = False) -> State:
    """
    Load an EEG recording into the pipeline state using the appropriate MNE reader.

    This step is the entry point for signal data in the preprocessing pipeline.
    It selects the correct MNE loader based on file extension, loads the file,
    forces the data into memory, and initializes the bad-channel list.

    Supported extensions
    --------------------
    .bdf, .edf, .cnt, .set, .fif, .mff, .egi, .raw, .gdf

    Expected params
    ---------------
    path : str or Path
        Path to the EEG recording file.

    preload : bool, optional
        Whether to preload the file during the MNE read step.
        Default is True.

    Returns
    -------
    state : dict
        Updated pipeline state containing:
        - state["raw"] : loaded MNE Raw object
        - state["loaded_eeg_path"] : resolved path to the loaded EEG file

    Raises
    ------
    FileNotFoundError
        If the path does not exist.

    ValueError
        If the file extension is unsupported.

    Notes
    -----
    - This function stores the resolved input path in `state["loaded_eeg_path"]`.
      That is useful for downstream ERP steps that need to match the currently
      loaded EEG file back to metadata or the original source file.
    - The function also initializes `raw.info["bads"] = []` so later cleaning
      steps can safely append bad channels.
    """
    path = params["path"]
    preload = params.get("preload", True)

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    ext = p.suffix.lower()

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

    if ext not in loaders:
        raise ValueError(
            f"Unsupported EEG extension: {ext}. "
            f"Supported: {sorted(loaders.keys())}"
        )

    if verbose:
        print(f"\nLoading EEG file: {p}")
        print(f"Detected type: {ext}")

    raw = loaders[ext](p, preload=preload, verbose=verbose)

    # Ensure data are fully loaded into memory for downstream processing.
    raw.load_data()

    # Initialize bad-channel bookkeeping for later cleaning steps.
    raw.info["bads"] = []

    # Store both the loaded signal and the source path in pipeline state.
    state["raw"] = raw
    state["loaded_eeg_path"] = str(p)
    return state



# def step_load_eeg(state: State, params: Params, verbose: bool = False) -> State:
#     """
#     Load an EEG file into the pipeline state using the correct MNE reader.

#     Expected params
#     ---------------
#     path : str or Path
#         File path to the EEG recording.
#     preload : bool, optional
#         Whether to preload the data into memory (default: True).
#     """
#     path = params["path"]
#     preload = params.get("preload", True)

#     p = Path(path).expanduser().resolve()
#     if not p.exists():
#         raise FileNotFoundError(f"Path does not exist: {p}")

#     ext = p.suffix.lower()

#     loaders = {
#         ".bdf": mne.io.read_raw_bdf,
#         ".edf": mne.io.read_raw_edf,
#         ".cnt": mne.io.read_raw_cnt,
#         ".set": mne.io.read_raw_eeglab,
#         ".fif": mne.io.read_raw_fif,
#         ".mff": mne.io.read_raw_egi,
#         ".egi": mne.io.read_raw_egi,
#         ".raw": mne.io.read_raw_egi,
#         ".gdf": mne.io.read_raw_gdf,
#     }

#     if ext not in loaders:
#         raise ValueError(
#             f"Unsupported EEG extension: {ext}. "
#             f"Supported: {sorted(loaders.keys())}"
#         )

#     if verbose:
#         print(f"\nLoading EEG file: {p}")
#         print(f"Detected type: {ext}")

#     raw = loaders[ext](p, preload=preload, verbose=verbose)

#     # Force full preload and initialize bad-channel list
#     raw.load_data()
#     raw.info["bads"] = []

#     state["raw"] = raw
#     return state



def step_scale_data(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Multiply raw data by a constant scale factor.

    Useful when EEG data were loaded in microvolts but MNE expects volts.

    Example params
    --------------
    {"factor": 1e-6}
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    factor = float(params.get("factor", 1.0))

    if verbose:
        print(f"→ Scaling raw data by factor: {factor}")

    raw.load_data()
    raw._data *= factor

    state["raw"] = raw
    state["scale_factor"] = factor
    return state



def _safe_high_cut_for_iclabel(sfreq: float, desired: float = 100.0) -> float:
    """
    Keep ICA/ICLabel high cutoff below Nyquist.
    """
    nyq = sfreq / 2.0
    return float(min(desired, nyq - 1.0)) if nyq > 2 else float(nyq * 0.8)


def step_run_ica_iclabel(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run ICA on a dedicated filtered copy of raw, classify ICs with ICLabel,
    exclude selected artifact ICs, and apply the ICA solution back to the
    original raw signal.

    Expected params
    ---------------
    ica : dict, optional
        Passed to mne.preprocessing.ICA(...)
        Example:
            {
                "n_components": None,
                "method": "infomax",
                "fit_params": {"extended": True},
                "random_state": 42,
                "max_iter": "auto"
            }

    fit : dict, optional
        Controls preprocessing of the ICA-fit branch.
        Example:
            {
                "notch_freqs": [60.0, 120.0],
                "l_freq": 1.0,
                "desired_h_freq": 100.0,
                "apply_average_ref": True,
                "picks": "eeg"
            }

    iclabel : dict, optional
        Controls ICLabel-based exclusion.
        Example:
            {
                "artifact_labels": [
                    "eye blink",
                    "muscle artifact",
                    "line noise",
                    "heart beat",
                    "channel noise"
                ],
                "prob_threshold": 0.7,
                "backend": "onnx"
            }

    store : dict, optional
        Keys to store outputs in state.
        Defaults:
            {
                "raw_key": "raw",
                "ica_key": "ica",
                "ic_df_key": "iclabel_df",
                "exclude_key": "excluded_ics",
                "labels_key": "ic_labels",
                "proba_key": "ic_probs"
            }

    Notes
    -----
    - This step removes artifact *components*, not noisy epochs.
    - ICA is fit on a filtered copy, then applied to the original raw.
    - Requires mne_icalabel to be installed and importable.
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    ica_kw = params.get("ica", {})
    fit_kw = params.get("fit", {})
    iclabel_kw = params.get("iclabel", {})
    store_kw = params.get("store", {})

    if not isinstance(ica_kw, dict) or not isinstance(fit_kw, dict) or not isinstance(iclabel_kw, dict):
        raise TypeError("step_run_ica_iclabel expects dicts for 'ica', 'fit', and 'iclabel'.")

    # -----------------------------
    # Defaults
    # -----------------------------
    ica_defaults = {
        "n_components": None,
        "method": "infomax",
        "fit_params": {"extended": True},
        "random_state": 42,
        "max_iter": "auto",
    }
    fit_defaults = {
        "notch_freqs": None,
        "l_freq": 1.0,
        "desired_h_freq": 100.0,
        "apply_average_ref": True,
        "picks": "eeg",
    }
    iclabel_defaults = {
        "artifact_labels": [
            "eye blink",
            "muscle artifact",
            "line noise",
            "heart beat",
            "channel noise",
        ],
        "prob_threshold": 0.7,
        "backend": "onnx",
    }

    ica_cfg = {**ica_defaults, **ica_kw}
    fit_cfg = {**fit_defaults, **fit_kw}
    iclabel_cfg = {**iclabel_defaults, **iclabel_kw}

    raw_key = store_kw.get("raw_key", "raw")
    ica_key = store_kw.get("ica_key", "ica")
    ic_df_key = store_kw.get("ic_df_key", "iclabel_df")
    exclude_key = store_kw.get("exclude_key", "excluded_ics")
    labels_key = store_kw.get("labels_key", "ic_labels")
    proba_key = store_kw.get("proba_key", "ic_probs")

    # -----------------------------
    # ICA fit branch
    # -----------------------------
    raw_ic = raw.copy().load_data()

    notch_freqs = fit_cfg.get("notch_freqs", None)
    if notch_freqs is not None and len(notch_freqs) > 0:
        if verbose:
            print(f"→ ICA branch notch filter: {notch_freqs}")
        raw_ic.notch_filter(freqs=list(notch_freqs), picks=fit_cfg["picks"], verbose=False)

    l_freq = float(fit_cfg.get("l_freq", 1.0))
    desired_h_freq = float(fit_cfg.get("desired_h_freq", 100.0))
    hi = _safe_high_cut_for_iclabel(raw_ic.info["sfreq"], desired=desired_h_freq)

    if verbose:
        print(f"→ ICA branch bandpass: {l_freq}–{hi} Hz")

    raw_ic.filter(
        l_freq=l_freq,
        h_freq=hi,
        picks=fit_cfg["picks"],
        verbose=False
    )

    if bool(fit_cfg.get("apply_average_ref", True)):
        if verbose:
            print("→ ICA branch average reference")
        raw_ic.set_eeg_reference("average", verbose=False)

    # -----------------------------
    # Fit ICA
    # -----------------------------
    if verbose:
        print(f"→ Fitting ICA with params: {ica_cfg}")

    ica = ICA(
        n_components=ica_cfg["n_components"],
        method=ica_cfg["method"],
        fit_params=ica_cfg["fit_params"],
        random_state=ica_cfg["random_state"],
        max_iter=ica_cfg["max_iter"],
    )
    ica.fit(raw_ic, picks=fit_cfg["picks"], verbose=False)

    # -----------------------------
    # ICLabel classification
    # -----------------------------
    backend = iclabel_cfg.get("backend", "onnx")
    labels_pred_proba = iclabel_label_components(
        raw_ic,
        ica,
        inplace=False,
        backend=backend,
    )

    labels_pred = np.argmax(labels_pred_proba, axis=1)
    labels = [
        ICALABEL_METHODS_NUMERICAL_TO_STRING["iclabel"][i]
        for i in labels_pred
    ]
    y_pred_proba = labels_pred_proba[np.arange(ica.n_components_), labels_pred]

    artifact_labels = {str(x).lower() for x in iclabel_cfg.get("artifact_labels", [])}
    prob_threshold = iclabel_cfg.get("prob_threshold", 0.7)

    exclude = []
    for i, (lab, p) in enumerate(zip(labels, y_pred_proba)):
        if lab.lower() in artifact_labels:
            if prob_threshold is None or float(p) >= float(prob_threshold):
                exclude.append(i)

    if verbose:
        print(f"→ Excluding ICs: {exclude}")

    # -----------------------------
    # Apply ICA back to original raw
    # -----------------------------
    ica.exclude = exclude
    raw_clean = raw.copy()
    raw_clean = ica.apply(raw_clean, verbose=False)

    # -----------------------------
    # Build IC summary table
    # -----------------------------
    ic_df = pd.DataFrame({
        "ic": np.arange(len(labels)),
        "label": labels,
        "y_pred_proba": y_pred_proba.astype(float),
        "excluded": [i in exclude for i in range(len(labels))],
    })

    # -----------------------------
    # Store outputs
    # -----------------------------
    state[raw_key] = raw_clean
    state[ica_key] = ica
    state[ic_df_key] = ic_df
    state[exclude_key] = exclude
    state[labels_key] = labels
    state[proba_key] = y_pred_proba.astype(float).tolist()

    return state



def step_prepare_erp_from_converted_df(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Prepare ERP events and event-locked epochs using the currently loaded EEG
    signal and the original FieldTrip `.mat` file referenced in converted_df.

    New capability
    --------------
    Optionally create a second event-locked Epochs object specifically for
    RANSAC cleaning via params["ransac_epochs"].

    Required params
    ---------------
    converted_df : pd.DataFrame

    source : dict
        - mat_col : str
        - data_name : str
        - events_key : str

    event_selection : dict
        - keep_values : sequence[int] | None
        - event_value_key : str
        - event_sample_key : str
        - collapse_to : int | None

    epochs : dict
        kwargs for final ERP epochs

    Optional params
    ---------------
    ransac_epochs : dict
        kwargs for temporary ERP epochs used by RANSAC

    store : dict
        Defaults:
        - events_key         -> "events_erp"
        - epochs_key         -> "epochs_erp"
        - event_counts_key   -> "event_counts_erp"
        - mat_events_key     -> "events_mat_raw"
        - ransac_epochs_key  -> "epochs_erp_ransac"
    """
    converted_df = params.get("converted_df")
    if converted_df is None or not isinstance(converted_df, pd.DataFrame):
        raise TypeError("prepare_erp_from_converted_df requires params['converted_df'] as a pandas DataFrame.")

    source_kw = params.get("source", {})
    select_kw = params.get("event_selection", {})
    epochs_kw = params.get("epochs", {})
    ransac_epochs_kw = params.get("ransac_epochs", None)
    store_kw = params.get("store", {})

    if not isinstance(source_kw, dict) or not isinstance(select_kw, dict) or not isinstance(epochs_kw, dict):
        raise TypeError("params['source'], params['event_selection'], and params['epochs'] must be dicts.")
    if ransac_epochs_kw is not None and not isinstance(ransac_epochs_kw, dict):
        raise TypeError("params['ransac_epochs'] must be a dict if provided.")

    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No state['raw'] found. Run load_eeg first.")

    loaded_eeg_path = state.get("loaded_eeg_path")
    if loaded_eeg_path is None:
        raise RuntimeError("No state['loaded_eeg_path'] found. Update step_load_eeg to store it.")

    loaded_eeg_path = Path(loaded_eeg_path).resolve()

    mat_col = source_kw.get("mat_col", "filepath")
    data_name = source_kw.get("data_name", "ft")
    events_field = source_kw.get("events_key", "events")

    if mat_col not in converted_df.columns:
        raise KeyError(f"Converted DataFrame is missing required mat path column '{mat_col}'.")
    if "filename" not in converted_df.columns:
        raise KeyError("Converted DataFrame must contain a 'filename' column for matching.")

    loaded_stem = loaded_eeg_path.stem
    loaded_stem = loaded_stem.replace("_eeg", "")

    df_work = converted_df.copy()
    df_work["_filename_stem"] = df_work["filename"].map(lambda x: Path(str(x)).stem)

    matches = df_work[df_work["_filename_stem"] == loaded_stem]
    if len(matches) == 0:
        raise RuntimeError(
            f"No matching row found in converted_df for loaded EEG file stem '{loaded_stem}'."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple matching rows found in converted_df for loaded EEG file stem '{loaded_stem}'."
        )

    row_obj = matches.iloc[0]

    mat_path = Path(row_obj[mat_col]).expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"Original .mat file not found: {mat_path}")

    mat = loadmat(mat_path, simplify_cells=True)
    if data_name not in mat:
        raise KeyError(f"Top-level MATLAB variable '{data_name}' not found in: {mat_path}")

    ft = mat[data_name]
    if not isinstance(ft, dict):
        raise TypeError(f"Expected '{data_name}' to load as dict-like, got: {type(ft)}")

    if events_field not in ft:
        raise KeyError(f"Field '{events_field}' not found inside MATLAB variable '{data_name}'.")

    mat_events = ft[events_field]
    if not isinstance(mat_events, list):
        raise TypeError(f"Expected ft['{events_field}'] to be a list, got: {type(mat_events)}")

    if "fsample" not in ft:
        raise KeyError(
            f"Field '{data_name}' is missing 'fsample'. This is required to map "
            "original event sample indices onto the current raw sampling grid."
        )

    orig_sfreq = float(ft["fsample"])
    current_sfreq = float(raw.info["sfreq"])
    sample_scale = current_sfreq / orig_sfreq

    value_key = select_kw.get("event_value_key", "value")
    sample_key = select_kw.get("event_sample_key", "sample")
    keep_values = select_kw.get("keep_values", None)
    collapse_to = select_kw.get("collapse_to", None)

    if keep_values is not None:
        keep_values = set(int(v) for v in keep_values)

    rows = []
    for ev in mat_events:
        if not isinstance(ev, dict):
            continue
        if value_key not in ev or sample_key not in ev:
            continue

        code = int(ev[value_key])
        sample_orig = int(ev[sample_key])

        if keep_values is not None and code not in keep_values:
            continue

        if collapse_to is not None:
            code = int(collapse_to)

        sample_current = int(round(sample_orig * sample_scale))
        rows.append([sample_current, 0, code])

    if len(rows) == 0:
        raise RuntimeError("No matching ERP events found after applying event_selection.")

    events = np.asarray(rows, dtype=int)

    n_times = int(raw.n_times)
    if np.any(events[:, 0] < 0) or np.any(events[:, 0] >= n_times):
        bad_mask = (events[:, 0] < 0) | (events[:, 0] >= n_times)
        n_bad = int(np.sum(bad_mask))
        first_bad = events[np.where(bad_mask)[0][0]].tolist()
        raise ValueError(
            "Some ERP event samples fall outside raw data bounds after sample-index mapping. "
            f"orig_sfreq={orig_sfreq}, current_sfreq={current_sfreq}, n_times={n_times}, "
            f"n_bad_events={n_bad}, first_bad_event={first_bad}"
        )

    unique_codes, counts = np.unique(events[:, 2], return_counts=True)
    event_counts = {int(k): int(v) for k, v in zip(unique_codes, counts)}

    ep_defaults = {
        "tmin": -0.2,
        "tmax": 0.5,
        "baseline": (None, 0),
        "preload": True,
        "reject_by_annotation": True,
        "event_id": None,
    }
    ep_kw = {**ep_defaults, **epochs_kw}
    epochs = mne.Epochs(raw, events, **ep_kw)

    ransac_epochs = None
    if ransac_epochs_kw is not None:
        ransac_defaults = {
            "tmin": -0.2,
            "tmax": 0.5,
            "baseline": (None, 0),
            "preload": True,
            "reject_by_annotation": True,
            "event_id": ep_kw.get("event_id", None),
        }
        ransac_ep_kw = {**ransac_defaults, **ransac_epochs_kw}
        ransac_epochs = mne.Epochs(raw, events, **ransac_ep_kw)

    events_key = store_kw.get("events_key", "events_erp")
    epochs_key = store_kw.get("epochs_key", "epochs_erp")
    counts_key = store_kw.get("event_counts_key", "event_counts_erp")
    mat_events_key = store_kw.get("mat_events_key", "events_mat_raw")
    ransac_epochs_key = store_kw.get("ransac_epochs_key", "epochs_erp_ransac")

    state[mat_events_key] = mat_events
    state[events_key] = events
    state[epochs_key] = epochs
    state[counts_key] = event_counts
    state["matched_mat_path"] = str(mat_path)
    state["erp_orig_sfreq"] = orig_sfreq
    state["erp_current_sfreq"] = current_sfreq
    state["erp_event_sample_scale"] = sample_scale

    if ransac_epochs is not None:
        state[ransac_epochs_key] = ransac_epochs

    if verbose:
        print(f"→ Loaded EEG path: {loaded_eeg_path}")
        print(f"→ Matched MAT path: {mat_path}")
        print(f"→ Original sfreq: {orig_sfreq}")
        print(f"→ Current raw sfreq: {current_sfreq}")
        print(f"→ Event sample scale: {sample_scale}")
        print(f"→ Event counts: {event_counts}")
        if ransac_epochs is not None:
            print(f"→ Stored RANSAC ERP epochs in state['{ransac_epochs_key}']")

    return state


# def step_prepare_erp_from_converted_df(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Prepare ERP events and event-locked epochs using the currently loaded EEG
#     signal and the original FieldTrip `.mat` file referenced in converted_df.

#     This step is intended for ERP workflows where:
#       - the EEG signal has already been loaded into state["raw"], typically from
#         a converted `.fif` file or directly from a `.mat` file, and
#       - the original event metadata still lives in the source FieldTrip `.mat`.

#     The function uses `state["loaded_eeg_path"]` to identify which EEG file is
#     currently being processed, matches that file to a row in `converted_df`,
#     retrieves the original `.mat` path from that row, loads FieldTrip events,
#     filters the requested event codes, rescales event sample indices if the
#     current raw sampling frequency differs from the original `.mat` sampling
#     frequency, and builds an `mne.Epochs` object.

#     Required params
#     ---------------
#     converted_df : pd.DataFrame
#         DataFrame linking processed EEG files back to their original source
#         files. It must contain:
#         - 'filename' : filename used for matching the current file
#         - source["mat_col"] : column containing original `.mat` filepaths

#     source : dict
#         Controls where ERP event metadata are read from.

#         Expected keys:
#         - mat_col : str
#             Column in converted_df containing the original `.mat` path.
#             Default: "filepath"
#         - data_name : str
#             Top-level MATLAB variable name inside the `.mat` file.
#             Default: "ft"
#         - events_key : str
#             Field inside the loaded MATLAB struct containing event data.
#             Default: "events"

#     event_selection : dict
#         Controls which events are kept and how they are interpreted.

#         Expected keys:
#         - keep_values : sequence[int] | None
#             Event codes to keep. If None, all events are kept.
#         - event_value_key : str
#             Event dict key containing the event code. Default: "value"
#         - event_sample_key : str
#             Event dict key containing the event sample index. Default: "sample"
#         - collapse_to : int | None
#             If provided, all kept events are reassigned to this one event code.

#     epochs : dict
#         Keyword arguments used to create `mne.Epochs`.
#         Common keys:
#         - tmin
#         - tmax
#         - baseline
#         - preload
#         - reject_by_annotation
#         - event_id

#     Optional params
#     ---------------
#     store : dict
#         Keys used to store outputs in state.

#         Defaults:
#         - events_key      -> "events_erp"
#         - epochs_key      -> "epochs_erp"
#         - event_counts_key-> "event_counts_erp"
#         - mat_events_key  -> "events_mat_raw"

#     Returns
#     -------
#     state : dict
#         Updated pipeline state containing:
#         - state["events_erp"] (or custom events_key)
#         - state["epochs_erp"] (or custom epochs_key)
#         - state["event_counts_erp"] (or custom event_counts_key)
#         - state["matched_mat_path"]

#     Raises
#     ------
#     RuntimeError
#         If state["raw"] or state["loaded_eeg_path"] is missing, or if no
#         matching row is found in converted_df.

#     KeyError
#         If required columns or MATLAB fields are missing.

#     FileNotFoundError
#         If the matched original `.mat` file does not exist.

#     ValueError
#         If event samples fall outside raw data bounds after sample-index mapping.

#     Notes
#     -----
#     This step does not reload the EEG signal by default. It assumes the EEG
#     currently in `state["raw"]` is the signal you want to epoch, and only
#     reloads the original `.mat` to recover event metadata.

#     Important
#     ---------
#     If the current raw has been resampled relative to the original `.mat`,
#     this function rescales the original event sample indices from the `.mat`
#     sampling grid onto the current raw sampling grid before creating epochs.
#     """
#     # -----------------------------
#     # 1) Validate top-level inputs
#     # -----------------------------
#     # Confirm that the metadata table was provided and is a DataFrame.
#     converted_df = params.get("converted_df")
#     if converted_df is None or not isinstance(converted_df, pd.DataFrame):
#         raise TypeError("prepare_erp_from_converted_df requires params['converted_df'] as a pandas DataFrame.")

#     # Pull nested parameter groups. This matches the style used elsewhere
#     # in your preprocessing pipeline.
#     source_kw = params.get("source", {})
#     select_kw = params.get("event_selection", {})
#     epochs_kw = params.get("epochs", {})
#     store_kw = params.get("store", {})

#     if not isinstance(source_kw, dict) or not isinstance(select_kw, dict) or not isinstance(epochs_kw, dict):
#         raise TypeError("params['source'], params['event_selection'], and params['epochs'] must be dicts.")

#     # -----------------------------
#     # 2) Validate required state
#     # -----------------------------
#     # The EEG signal itself should already be loaded by step_load_eeg.
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No state['raw'] found. Run load_eeg first.")

#     # We also need the path of the currently loaded EEG file so we can match
#     # it back to the correct row of converted_df.
#     loaded_eeg_path = state.get("loaded_eeg_path")
#     if loaded_eeg_path is None:
#         raise RuntimeError("No state['loaded_eeg_path'] found. Update step_load_eeg to store it.")

#     loaded_eeg_path = Path(loaded_eeg_path).resolve()

#     # -----------------------------
#     # 3) Read source settings
#     # -----------------------------
#     # These control where the original ERP metadata are found inside converted_df
#     # and inside the MATLAB file.
#     mat_col = source_kw.get("mat_col", "filepath")
#     data_name = source_kw.get("data_name", "ft")
#     events_field = source_kw.get("events_key", "events")

#     if mat_col not in converted_df.columns:
#         raise KeyError(f"Converted DataFrame is missing required mat path column '{mat_col}'.")

#     # -----------------------------
#     # 4) Match current EEG file to converted_df
#     # -----------------------------
#     # We match by filename stem so that the currently loaded EEG file can be
#     # linked back to the row that contains the original .mat filepath.
#     loaded_stem = loaded_eeg_path.stem
#     loaded_stem = loaded_stem.replace("_eeg", "")

#     df_work = converted_df.copy()
#     df_work["_filename_stem"] = df_work["filename"].map(lambda x: Path(str(x)).stem)

#     matches = df_work[df_work["_filename_stem"] == loaded_stem]
#     if len(matches) == 0:
#         raise RuntimeError(
#             f"No matching row found in converted_df for loaded EEG file stem '{loaded_stem}'."
#         )
#     if len(matches) > 1:
#         raise RuntimeError(
#             f"Multiple matching rows found in converted_df for loaded EEG file stem '{loaded_stem}'."
#         )

#     row_obj = matches.iloc[0]

#     # -----------------------------
#     # 5) Resolve and validate .mat path
#     # -----------------------------
#     # The matched row tells us which original FieldTrip .mat file contains
#     # the ERP event metadata for the current EEG recording.
#     mat_path = Path(row_obj[mat_col]).expanduser().resolve()
#     if not mat_path.exists():
#         raise FileNotFoundError(f"Original .mat file not found: {mat_path}")

#     # -----------------------------
#     # 6) Load original MATLAB event metadata
#     # -----------------------------
#     # Read the original FieldTrip structure from the .mat and locate the
#     # field containing the event list.
#     mat = loadmat(mat_path, simplify_cells=True)
#     if data_name not in mat:
#         raise KeyError(f"Top-level MATLAB variable '{data_name}' not found in: {mat_path}")

#     ft = mat[data_name]
#     if not isinstance(ft, dict):
#         raise TypeError(f"Expected '{data_name}' to load as dict-like, got: {type(ft)}")

#     if events_field not in ft:
#         raise KeyError(f"Field '{events_field}' not found inside MATLAB variable '{data_name}'.")

#     mat_events = ft[events_field]
#     if not isinstance(mat_events, list):
#         raise TypeError(f"Expected ft['{events_field}'] to be a list, got: {type(mat_events)}")

#     # -----------------------------
#     # 7) Resolve original vs current sampling rate
#     # -----------------------------
#     # Event sample indices in the FieldTrip event list are expressed on the
#     # original .mat sample grid. If the current raw has been resampled, we
#     # must map those original sample indices onto the current raw sample grid.
#     if "fsample" not in ft:
#         raise KeyError(
#             f"Field '{data_name}' is missing 'fsample'. This is required to map "
#             "original event sample indices onto the current raw sampling grid."
#         )

#     orig_sfreq = float(ft["fsample"])
#     current_sfreq = float(raw.info["sfreq"])
#     sample_scale = current_sfreq / orig_sfreq

#     # -----------------------------
#     # 8) Build an MNE-compatible events array
#     # -----------------------------
#     # Convert the list of MATLAB event dicts into the standard MNE events array
#     # format: [sample, 0, event_code].
#     #
#     # Important: sample indices are taken from the original .mat event table
#     # and rescaled if the raw has been resampled.
#     value_key = select_kw.get("event_value_key", "value")
#     sample_key = select_kw.get("event_sample_key", "sample")
#     keep_values = select_kw.get("keep_values", None)
#     collapse_to = select_kw.get("collapse_to", None)

#     if keep_values is not None:
#         keep_values = set(int(v) for v in keep_values)

#     rows = []
#     for ev in mat_events:
#         if not isinstance(ev, dict):
#             continue
#         if value_key not in ev or sample_key not in ev:
#             continue

#         code = int(ev[value_key])
#         sample_orig = int(ev[sample_key])

#         # Keep only requested ERP event codes, if filtering is enabled.
#         if keep_values is not None and code not in keep_values:
#             continue

#         # Optionally collapse all selected events into a single code.
#         if collapse_to is not None:
#             code = int(collapse_to)

#         # Map the original sample index onto the current raw sampling grid.
#         # If no resampling occurred, this leaves the index unchanged.
#         sample_current = int(round(sample_orig * sample_scale))

#         rows.append([sample_current, 0, code])

#     if len(rows) == 0:
#         raise RuntimeError("No matching ERP events found after applying event_selection.")

#     events = np.asarray(rows, dtype=int)

#     # -----------------------------
#     # 9) Validate event sample bounds
#     # -----------------------------
#     # Make sure event sample indices actually fall within the currently loaded
#     # EEG recording after mapping to the current sampling grid.
#     n_times = int(raw.n_times)
#     if np.any(events[:, 0] < 0) or np.any(events[:, 0] >= n_times):
#         bad_mask = (events[:, 0] < 0) | (events[:, 0] >= n_times)
#         n_bad = int(np.sum(bad_mask))
#         first_bad = events[np.where(bad_mask)[0][0]].tolist()
#         raise ValueError(
#             "Some ERP event samples fall outside raw data bounds after sample-index mapping. "
#             f"orig_sfreq={orig_sfreq}, current_sfreq={current_sfreq}, n_times={n_times}, "
#             f"n_bad_events={n_bad}, first_bad_event={first_bad}"
#         )

#     # Also summarize counts per event code for quick QC.
#     unique_codes, counts = np.unique(events[:, 2], return_counts=True)
#     event_counts = {int(k): int(v) for k, v in zip(unique_codes, counts)}

#     # -----------------------------
#     # 10) Epoch the data
#     # -----------------------------
#     # Use the currently loaded EEG signal together with the recovered ERP events
#     # to create event-locked MNE epochs.
#     ep_defaults = {
#         "tmin": -0.2,
#         "tmax": 0.5,
#         "baseline": (None, 0),
#         "preload": True,
#         "reject_by_annotation": True,
#         "event_id": None,
#     }
#     ep_kw = {**ep_defaults, **epochs_kw}

#     epochs = mne.Epochs(raw, events, **ep_kw)

#     # -----------------------------
#     # 11) Resolve output state keys
#     # -----------------------------
#     # Allow users to customize where outputs are stored in the pipeline state.
#     events_key = store_kw.get("events_key", "events_erp")
#     epochs_key = store_kw.get("epochs_key", "epochs_erp")
#     counts_key = store_kw.get("event_counts_key", "event_counts_erp")
#     mat_events_key = store_kw.get("mat_events_key", "events_mat_raw")

#     # -----------------------------
#     # 12) Store outputs into state
#     # -----------------------------
#     # Save raw MATLAB events, MNE events array, epoch object, event counts,
#     # source path, and sampling-rate mapping details for later steps or QC.
#     state[mat_events_key] = mat_events
#     state[events_key] = events
#     state[epochs_key] = epochs
#     state[counts_key] = event_counts
#     state["matched_mat_path"] = str(mat_path)
#     state["erp_orig_sfreq"] = orig_sfreq
#     state["erp_current_sfreq"] = current_sfreq
#     state["erp_event_sample_scale"] = sample_scale

#     # -----------------------------
#     # 13) Optional verbose logging
#     # -----------------------------
#     if verbose:
#         print(f"→ Loaded EEG path: {loaded_eeg_path}")
#         print(f"→ Matched MAT path: {mat_path}")
#         print(f"→ Original sfreq: {orig_sfreq}")
#         print(f"→ Current raw sfreq: {current_sfreq}")
#         print(f"→ Event sample scale: {sample_scale}")
#         print(f"→ Event counts: {event_counts}")

#     return state




def step_set_montage(state: State, params: Params, verbose: bool = False) -> State:
    """
    Set a standard EEG montage on the current Raw object.
    Params
    ------
    kind : str
        Montage name (e.g., "standard_1020", "biosemi64").
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    kind = params["kind"]
    montage = mne.channels.make_standard_montage(kind)
    raw.set_montage(montage, on_missing="ignore",)

    if verbose:
        print(f"→ Set montage: {kind}")

    state["raw"] = raw
    return state


def step_drop_channels(state: State, params: Params, verbose: bool = False) -> State:
    """
    Drop specified channels from the Raw object in state.

    Params
    ------
    names : list[str]
        Channel names to remove (e.g., ["M1", "M2"]).
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    names = params.get("names", [])
    if not names:
        return state

    present = [ch for ch in names if ch in raw.ch_names]
    missing = [ch for ch in names if ch not in raw.ch_names]

    if verbose:
        print(f"→ Dropping channels: {present}")
        if missing:
            print(f"→ (not found, skipping): {missing}")

    if present:
        raw.drop_channels(present)

    state["raw"] = raw
    return state



def step_demote_unlocalized(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Demote EEG channels with missing/invalid XYZ coordinates to 'misc'.

    Optional params:
      store_key: str  (default: "demoted_unlocalized")
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    eeg_inds = mne.pick_types(
        raw.info,
        eeg=True, meg=False, eog=False, stim=False, misc=False,
        exclude=[]
    )

    demoted = []
    for idx in eeg_inds:
        loc = np.asarray(raw.info["chs"][idx]["loc"][:3], float)
        if not (np.isfinite(loc).all() and not np.allclose(loc, 0.0)):
            ch_name = raw.info["chs"][idx]["ch_name"]
            demoted.append(ch_name)

    if demoted:
        raw.set_channel_types({ch: "misc" for ch in demoted})
        if verbose:
            print(f"Demoted EEG→misc (missing/invalid xyz): {demoted}")
    else:
        if verbose:
            print("→ No unlocalized EEG channels found to demote.")

    # Store results (handy for debugging / later steps)
    store_key = params.get("store_key", "demoted_unlocalized")
    state[store_key] = demoted

    state["raw"] = raw
    return state

def step_bandpass(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Apply band-pass filtering to the current Raw object in state.

    Example params:
      {"l_freq": 0.5, "h_freq": 45.0, "phase": "zero", "fir_design": "firwin"}
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    raw.filter(**params) 


    state["raw"] = raw
    return state

def step_resample_eeg(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Resample the current Raw object in state.

    Required params
    ---------------
    sfreq : float
        New sampling frequency in Hz.
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    if verbose:
        print(f"→ Resample with params: {params}")

    raw.resample(**params)  # let MNE validate + raise
    state["raw"] = raw
    return state


def step_notch_filter(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Apply notch filtering to the current Raw object in state.
    Requires params["freqs"] to be < Nyquist (sfreq / 2).
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    # --- Hard validation: freqs must be below Nyquist ---
    if "freqs" not in params:
        raise ValueError("notch_filter requires 'freqs' in params (e.g., {'freqs': [60, 120]}).")

    freqs = params["freqs"]
    if isinstance(freqs, (int, float)):
        freqs_list = [float(freqs)]
    else:
        freqs_list = [float(f) for f in freqs]

    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    bad = [f for f in freqs_list if f >= nyq]
    if bad:
        msg = (
            f"Invalid notch freqs {bad}: must be < Nyquist ({nyq:.2f} Hz) "
            f"given current sfreq={sfreq:.2f} Hz. "
            f"Resample higher or choose lower freqs."
        )
        print(msg)
        raise ValueError(msg)

    if verbose:
        print(f"→ Notch filter with params: {params}")

    raw.notch_filter(**params)  # let MNE validate remaining details
    state["raw"] = raw
    return state


def step_plot_raw(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Plot the raw time-series browser using MNE: raw.plot(**params)

    Example params:
      {"n_channels": 32, "picks": "eeg", "duration": 10.0}
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    if verbose:
        print(f"→ raw.plot with params: {params}")

    raw.plot(**params)
    return state




def step_plot_psd(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute and plot PSD in one shot:
        raw.compute_psd(**params).plot()
        plt.show()

    params are passed ONLY to raw.compute_psd.
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    if verbose:
        print(f"→ raw.compute_psd with params: {params}")

    raw.compute_psd(**params).plot()
    plt.show()

    return state



def step_ransac_clean(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    RANSAC cleaning with two modes:

    Mode A: use precomputed epochs from state (recommended for ERP)
      params = {
        "use_state_epochs": True,
        "state_epochs_key": "epochs_erp_ransac",
        "ransac": {"n_jobs": -1},
        "reset_bads": True
      }

    Mode B: create fixed-length epochs internally (legacy behavior)
      params = {
        "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
        "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, ...},
        "ransac": {"n_jobs": -1},
        "reset_bads": True
      }
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    use_state_epochs = params.get("use_state_epochs", False)
    state_epochs_key = params.get("state_epochs_key", "epochs_erp_ransac")
    ransac_kw = params.get("ransac", {})
    reset_bads = params.get("reset_bads", True)

    if not isinstance(ransac_kw, dict):
        raise TypeError("ransac_clean expects params['ransac'] to be a dict.")

    if use_state_epochs:
        epochs = state.get(state_epochs_key)
        if epochs is None:
            raise RuntimeError(
                f"RANSAC requested use_state_epochs=True, but state['{state_epochs_key}'] was not found."
            )
        if verbose:
            print(f"→ Using precomputed epochs from state['{state_epochs_key}'] for RANSAC")
    else:
        events_kw = params.get("events", {})
        epochs_kw = params.get("epochs", {})

        if not isinstance(events_kw, dict) or not isinstance(epochs_kw, dict):
            raise TypeError("ransac_clean expects dicts for 'events' and 'epochs' when use_state_epochs=False.")

        if "duration" not in events_kw:
            raise ValueError("ransac_clean requires params['events']['duration'] when use_state_epochs=False.")

        if verbose:
            print(f"→ RANSAC events params: {events_kw}")

        events = mne.make_fixed_length_events(raw, **events_kw)

        if "tmax" not in epochs_kw:
            epochs_kw = dict(epochs_kw)
            epochs_kw["tmax"] = float(events_kw["duration"])

        if verbose:
            print(f"→ RANSAC epochs params: {epochs_kw}")

        epochs = mne.Epochs(raw, events, **epochs_kw)
        state["events_ransac"] = events
        state["epochs_ransac"] = epochs

    eeg_inds = mne.pick_types(
        epochs.info,
        eeg=True, meg=False, eog=False, stim=False, misc=False,
        exclude=[]
    )

    valid_picks = []
    for idx in eeg_inds:
        loc = np.asarray(epochs.info["chs"][idx]["loc"][:3], float)
        if np.isfinite(loc).all() and not np.allclose(loc, 0.0):
            valid_picks.append(idx)

    if len(valid_picks) == 0:
        raise RuntimeError(
            "No valid EEG channels with xyz locations found for RANSAC. "
            "Did you set a montage and/or demote unlocalized EEG?"
        )

    if "picks" not in ransac_kw:
        ransac_kw = dict(ransac_kw)
        ransac_kw["picks"] = valid_picks

    if verbose:
        print(f"→ RANSAC params: {ransac_kw}")

    ransac = Ransac(**ransac_kw)
    _ = ransac.fit_transform(epochs)
    bad_channels = list(getattr(ransac, "bad_chs_", []))

    if verbose:
        print(f"→ RANSAC bad channels: {bad_channels}")
        print("→ Interpolating bad channels on raw...")

    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=reset_bads)

    state["bad_channels"] = bad_channels
    state["raw"] = raw
    return state

# def step_ransac_clean(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     One-step RANSAC cleaning:
#       - fixed-length events
#       - epochs
#       - RANSAC bad-channel detection
#       - interpolate bad channels on Raw

#     Expected params (nested; keeps it clean):
#       {
#         "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
#         "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, ...},
#         "ransac": {"n_jobs": -1, ...},
#         "reset_bads": True   # optional (default True)
#       }
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     events_kw = params.get("events", {})
#     epochs_kw = params.get("epochs", {})
#     ransac_kw = params.get("ransac", {})
#     reset_bads = params.get("reset_bads", True)

#     if not isinstance(events_kw, dict) or not isinstance(epochs_kw, dict) or not isinstance(ransac_kw, dict):
#         raise TypeError("ransac_clean expects dicts for 'events', 'epochs', and 'ransac'.")

#     # ---- 1) Make fixed-length events ----
#     if "duration" not in events_kw:
#         raise ValueError("ransac_clean requires params['events']['duration'].")

#     if verbose:
#         print(f"→ RANSAC events params: {events_kw}")

#     events = mne.make_fixed_length_events(raw, **events_kw)

#     # ---- 2) Epoch the data ----
#     # Provide a sensible default tmax if user omitted it
#     if "tmax" not in epochs_kw:
#         epochs_kw = dict(epochs_kw)
#         epochs_kw["tmax"] = float(events_kw["duration"])

#     if verbose:
#         print(f"→ RANSAC epochs params: {epochs_kw}")

#     epochs = mne.Epochs(raw, events, **epochs_kw)

#     # ---- 3) Build valid picks (EEG channels with real xyz) ----
#     eeg_inds = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, misc=False, exclude=[])
#     valid_picks = []
#     for idx in eeg_inds:
#         loc = np.asarray(epochs.info["chs"][idx]["loc"][:3], float)
#         if np.isfinite(loc).all() and not np.allclose(loc, 0.0):
#             valid_picks.append(idx)

#     if len(valid_picks) == 0:
#         raise RuntimeError(
#             "No valid EEG channels with xyz locations found for RANSAC. "
#             "Did you set a montage and/or demote unlocalized EEG?"
#         )

#     # If user didn’t provide picks, use valid localized EEG by default
#     if "picks" not in ransac_kw:
#         ransac_kw = dict(ransac_kw)
#         ransac_kw["picks"] = valid_picks

#     if verbose:
#         print(f"→ RANSAC params: {ransac_kw}")

#     # ---- 4) Run RANSAC and interpolate bad channels ----
#     ransac = Ransac(**ransac_kw)
#     _ = ransac.fit_transform(epochs)
#     bad_channels = list(getattr(ransac, "bad_chs_", []))

#     if verbose:
#         print(f"→ RANSAC bad channels: {bad_channels}")
#         print("→ Interpolating bad channels on raw...")

#     raw.info["bads"] = bad_channels
#     raw.interpolate_bads(reset_bads=reset_bads)

#     # Store useful outputs (lightweight, but very handy)
#     state["events_ransac"] = events
#     state["epochs_ransac"] = epochs
#     state["bad_channels"] = bad_channels
#     state["raw"] = raw
#     return state



def step_set_reference(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Re-reference EEG channels using MNE.

    Example params:
      {"ref_channels": "average"}
      {"ref_channels": ["M1", "M2"], "projection": False}
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    if verbose:
        print(f"→ set_eeg_reference with params: {params}")

    out = raw.set_eeg_reference(**params)

    # MNE may return either raw or (raw, ref_data)
    if isinstance(out, tuple):
        raw, ref_data = out
        state["ref_data"] = ref_data
    else:
        raw = out

    state["raw"] = raw
    return state




def step_apply_csd(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Apply Current Source Density (surface Laplacian).

    Requires a montage with valid xyz locations.
    Params are passed directly to compute_current_source_density.
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    if verbose:
        print(f"→ Applying CSD with params: {params}")

    raw_csd = compute_current_source_density(raw, **params)

    state["raw"] = raw_csd
    state["csd_applied"] = True
    return state




def step_fixed_length_epochs(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Create fixed-length events and epochs from the current Raw.

    Params (recommended structure)
    ------------------------------
    {
      "events": {...},   # kwargs for mne.make_fixed_length_events(raw, **events)
      "epochs": {...},   # kwargs for mne.Epochs(raw, events, **epochs)
      "store": {         # optional
          "events_key": "events",
          "epochs_key": "epochs"
      }
    }

    Minimal example
    ---------------
    {
      "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
      "epochs": {"event_id": {"seg": 2}, "tmin": 0.0, "tmax": 2.0, "baseline": None,
                "reject": None, "detrend": 0, "preload": True}
    }
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    events_kw = params.get("events", {})
    epochs_kw = params.get("epochs", {})
    store_kw = params.get("store", {})

    if not isinstance(events_kw, dict) or not isinstance(epochs_kw, dict):
        raise TypeError("fixed_length_epochs expects dicts for params['events'] and params['epochs'].")

    # ---- events ----
    if "duration" not in events_kw:
        raise ValueError("fixed_length_epochs requires params['events']['duration'].")

    if verbose:
        print(f"→ Fixed-length events params: {events_kw}")

    events = mne.make_fixed_length_events(raw, **events_kw)

    # ---- epochs ----
    # Sensible default: if user omitted tmax, use duration
    if "tmax" not in epochs_kw:
        epochs_kw = dict(epochs_kw)
        epochs_kw["tmax"] = float(events_kw["duration"])

    if verbose:
        print(f"→ Epochs params: {epochs_kw}")

    epochs = mne.Epochs(raw, events, **epochs_kw)

    # ---- store ----
    events_key = store_kw.get("events_key", "events")
    epochs_key = store_kw.get("epochs_key", "epochs")

    state[events_key] = events
    state[epochs_key] = epochs

    if verbose:
        print(f"→ Stored events in state['{events_key}'], epochs in state['{epochs_key}']")

    return state


def _build_ops() -> Dict[str, StepFn]:
    # Maps step names used in config -> the function that executes that step
    return {
        "load_eeg": step_load_eeg,                      # Load EEG file into state["raw"]
        "scale_data": step_scale_data,                  # Scale data from uV to V
        "set_montage": step_set_montage,                # Attach electrode positions (montage) to raw
        "drop_channels": step_drop_channels,            # Remove unwanted channels (e.g., M1/M2)
        "demote_unlocalized": step_demote_unlocalized,  # Convert EEG chans with missing xyz to 'misc'
        "resample_eeg": step_resample_eeg,              # Change sampling rate (updates raw.info["sfreq"])
        "bandpass_filter": step_bandpass,               # Band-pass filter raw (e.g., 0.5–45 Hz)
        "notch_filter": step_notch_filter,              # Notch filter line noise (e.g., 60/120 Hz)
        "plot_raw": step_plot_raw,                      # Plot time-series browser (raw.plot)
        "plot_psd": step_plot_psd,                      # Plot PSD (raw.compute_psd(...).plot())
        "ransac_clean": step_ransac_clean,              # RANSAC bad-channel detection + interpolation
        "set_reference": step_set_reference,            # Re-reference EEG (e.g., average reference)
        "apply_csd": step_apply_csd,                    # Apply CSD / surface Laplacian (spatial sharpening)
        "fixed_length_epochs": step_fixed_length_epochs, # Create fixed-length events + epochs (final segments)
        "prepare_erp_from_converted_df": step_prepare_erp_from_converted_df,  # Extract erp time points and make epochs
        "run_ica_iclabel": step_run_ica_iclabel,         # ICA run
    }









def _parse_step_spec(spec: Any) -> tuple[Params, bool]:
    """
    Accept either:
      - short form: {"drop_channels": {"names": [...]}}
      - envelope:   {"drop_channels": {"params": {...}, "verbose": True}}
      - null:       {"some_step": None}  -> params={}, verbose=False
    """
    if spec is None:
        return {}, False
    if not isinstance(spec, dict):
        raise TypeError(f"Step spec must be a dict or None, got: {type(spec)}")

    # Envelope form
    if "params" in spec or "verbose" in spec:
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be a dict, got: {type(params)}")
        step_verbose = bool(spec.get("verbose", False))
        return params, step_verbose

    # Short form (params directly)
    return spec, False


def eeg_preprocess_pipeline(config: Dict[str, Any], ops: Optional[Mapping[str, StepFn]] = None) -> State:
    """
    Execute a sequence of steps described by:
      config = {"steps": [ {"load_eeg": {...}}, {"drop_channels": {...}}, ... ]}
    """
    if ops is None:
        ops = _build_ops()

    steps = config.get("steps")
    if not isinstance(steps, list):
        raise TypeError("config['steps'] must be a list")

    state: State = {}

    for idx, step_obj in enumerate(steps, start=1):
        if not isinstance(step_obj, dict) or len(step_obj) != 1:
            raise ValueError(
                f"Each step must be a single-key dict like {{'op': {{...}}}}. "
                f"Got at step {idx}: {step_obj!r}"
            )

        op_name, spec = next(iter(step_obj.items()))
        fn = ops.get(op_name)
        if fn is None:
            raise KeyError(
                f"Unknown step '{op_name}' at step {idx}. "
                f"Available: {sorted(ops.keys())}"
            )

        params, step_verbose = _parse_step_spec(spec)

        if step_verbose:
            print("=" * 100)
            print(f"STEP {idx}/{len(steps)} — {op_name}")
            print("=" * 100)
            if params:
                print(f"params: {params}")

        state = fn(state, params, verbose=step_verbose)
        if not isinstance(state, dict):
            raise TypeError(f"Step '{op_name}' must return a dict state.")
        
        
    print("\n✅ Preprocessing complete.")

    return state





config_nehemiah = config = {
    "steps": [
        # 1) Load EEG file
        {"load_eeg": {"params": {"path": "EEG1.bdf"}, "verbose": True}},

        # 2) Assign scalp electrode positions
        {"set_montage": {"params": {"kind": "biosemi64"}, "verbose": True}},

        # 3) Resample early (affects Nyquist + speeds later filtering)
        {"resample_eeg": {"params": {"sfreq": 250.0}, "verbose": True}},

        # 4) Quick QC: raw view
        {"plot_raw": {"params": {"n_channels": 32, "picks": "eeg"}, "verbose": True}},

        # 5) Quick QC: PSD
        {"plot_psd": {"params": {"picks": "eeg", "average": False}, "verbose": True}},

        # 6) Drop mastoids early
        {"drop_channels": {"params": {"names": ["M1", "M2"]}, "verbose": True}},

        # 7) Mark channels without valid xyz as 'misc' (prevents spatial ops issues)
        {"demote_unlocalized": {"params": {}, "verbose": True}},

        # 8) Band-pass filter
        {"bandpass_filter": {"params": {"l_freq": 0.5, "h_freq": 45.0, "phase": "zero", "fir_design": "firwin"}, "verbose": True}},

        # 9) Notch filter (line noise) — ensure freqs < Nyquist
        {"notch_filter": {"params": {"freqs": [60.0, 120.0], "phase": "zero", "filter_length": "auto"}, "verbose": True}},

        # 10) QC again: raw
        {"plot_raw": {"params": {"n_channels": 32, "picks": "eeg"}, "verbose": True}},

        # 11) QC again: PSD
        {"plot_psd": {"params": {"picks": "eeg", "average": False}, "verbose": True}},

        # 12) RANSAC bad-channel detection + interpolation
        {"ransac_clean": {"params": {
            "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
            "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, "baseline": (0, 0),
                       "reject": None, "detrend": 0, "preload": True, "verbose": False},
            "ransac": {"n_jobs": -1},
            "reset_bads": True
        }, "verbose": True}},

        # 13) Average reference
        {"set_reference": {"params": {"ref_channels": "average"}, "verbose": True}},

        # 14) CSD (surface Laplacian)
        {"apply_csd": {"params": {}, "verbose": True}},

        # 15) Final fixed-length epochs for ML
        {"fixed_length_epochs": {"params": {
            "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
            "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, "baseline": None,
                       "reject": None, "detrend": 0, "preload": True, "verbose": False},
            "store": {"events_key": "events_final", "epochs_key": "epochs_final"}
        }, "verbose": True}},

        # 16) Final PSD check (CSD’d data). Note "picks": "data" after doing PSD since 'eeg' gets replaced
        {"plot_psd": {"params": {"picks": "data", "average": False}, "verbose": True}},
    ]
}





def plot_pipeline_text(
    config,
    n_cols: int = 4,
    figsize: tuple = (14, 6),
    max_width: int = 28,
    line_height: float = 0.035,
):
    """
    Visualize the preprocessing pipeline as wrapped text blocks in columns.

    Parameters
    ----------
    config : dict
        Pipeline configuration containing a "steps" list, where each step is
        a dict like {"op_name": {params}}.
    n_cols : int, optional
        Number of text columns to use in the figure.
    figsize : tuple, optional
        Matplotlib figure size (width, height) in inches.
    max_width : int, optional
        Maximum character width for wrapping each "key = value" line.
    line_height : float, optional
        Vertical spacing per text line in Axes (0–1) coordinates.

    Notes
    -----
    - This is a convenience visualization: it does not modify the config.
    - Steps are laid out in reading order, top-to-bottom within each column,
      then left-to-right across columns.
    """

    steps = config.get("steps", [])
    if not steps:
        raise ValueError("config['steps'] is empty; nothing to plot.")

    n_steps = len(steps)
    # Ceiling division to decide how many steps go into each column
    steps_per_col = (n_steps + n_cols - 1) // n_cols

    # ------------------------------------------------------------------
    # Prepare formatted text blocks and count their line usage
    # ------------------------------------------------------------------
    formatted_blocks = []
    block_line_counts = []

    for i, step in enumerate(steps, start=1):
        # Each step is assumed to be {"op_name": {params}}
        op_name, op_params = next(iter(step.items()))

        # Wrap parameters into multiple lines, if necessary
        param_lines = []
        if op_params:
            for k, v in op_params.items():
                # Wrap "k = v" to max_width characters, indent continuation
                wrapped = textwrap.fill(
                    f"{k} = {v}",
                    width=max_width,
                    subsequent_indent=" " * 6,
                )
                # Indent the whole param block under the step label
                param_lines.append("    " + wrapped)

        # Build the final text block for this step
        if param_lines:
            block = f"{i}. {op_name}\n" + "\n".join(param_lines)
        else:
            block = f"{i}. {op_name}"

        formatted_blocks.append(block)
        # Count number of lines in this block for vertical spacing
        block_line_counts.append(block.count("\n") + 1)

    # ------------------------------------------------------------------
    # Set up figure and basic layout
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")  # hide axes frame and ticks

    # Compute x-coordinates for each column (normalized Axes coordinates)
    if n_cols == 1:
        x_positions = [0.1]
    else:
        x_positions = [
            0.1 + j * (0.8 / (n_cols - 1))  # spread columns across 80% width
            for j in range(n_cols)
        ]

    y_top = 0.9  # starting y-position (top) for each column

    # ------------------------------------------------------------------
    # Draw columns of text blocks
    # ------------------------------------------------------------------
    for col_idx in range(n_cols):
        # Determine which steps belong to this column
        start = col_idx * steps_per_col
        end = min((col_idx + 1) * steps_per_col, n_steps)

        y = y_top

        for idx in range(start, end):
            block = formatted_blocks[idx]
            num_lines = block_line_counts[idx]

            # Place this block at (x, y); anchor at top-left
            ax.text(
                x_positions[col_idx],
                y,
                block,
                fontsize=10,
                va="top",
                ha="left",
                family="monospace",
            )

            # Move y down by the block height (lines * line_height) plus padding
            y -= num_lines * line_height + 0.03

    ax.set_title("EEG Preprocessing Pipeline", fontsize=14)
    plt.show()

    