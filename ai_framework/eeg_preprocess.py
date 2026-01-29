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

State = Dict[str, Any]
Params = Dict[str, Any]
StepFn = Callable[[State, Params, bool], State]


# ----------------------------
# Step functions (your style)
# ----------------------------

def step_load_eeg(state: State, params: Params, verbose: bool = False) -> State:
    """
    Load an EEG file into the pipeline state using the correct MNE reader.

    Expected params
    ---------------
    path : str or Path
        File path to the EEG recording.
    preload : bool, optional
        Whether to preload the data into memory (default: True).
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

    # Force full preload and initialize bad-channel list
    raw.load_data()
    raw.info["bads"] = []

    state["raw"] = raw
    return state


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
    One-step RANSAC cleaning:
      - fixed-length events
      - epochs
      - RANSAC bad-channel detection
      - interpolate bad channels on Raw

    Expected params (nested; keeps it clean):
      {
        "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
        "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, ...},
        "ransac": {"n_jobs": -1, ...},
        "reset_bads": True   # optional (default True)
      }
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    events_kw = params.get("events", {})
    epochs_kw = params.get("epochs", {})
    ransac_kw = params.get("ransac", {})
    reset_bads = params.get("reset_bads", True)

    if not isinstance(events_kw, dict) or not isinstance(epochs_kw, dict) or not isinstance(ransac_kw, dict):
        raise TypeError("ransac_clean expects dicts for 'events', 'epochs', and 'ransac'.")

    # ---- 1) Make fixed-length events ----
    if "duration" not in events_kw:
        raise ValueError("ransac_clean requires params['events']['duration'].")

    if verbose:
        print(f"→ RANSAC events params: {events_kw}")

    events = mne.make_fixed_length_events(raw, **events_kw)

    # ---- 2) Epoch the data ----
    # Provide a sensible default tmax if user omitted it
    if "tmax" not in epochs_kw:
        epochs_kw = dict(epochs_kw)
        epochs_kw["tmax"] = float(events_kw["duration"])

    if verbose:
        print(f"→ RANSAC epochs params: {epochs_kw}")

    epochs = mne.Epochs(raw, events, **epochs_kw)

    # ---- 3) Build valid picks (EEG channels with real xyz) ----
    eeg_inds = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, misc=False, exclude=[])
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

    # If user didn’t provide picks, use valid localized EEG by default
    if "picks" not in ransac_kw:
        ransac_kw = dict(ransac_kw)
        ransac_kw["picks"] = valid_picks

    if verbose:
        print(f"→ RANSAC params: {ransac_kw}")

    # ---- 4) Run RANSAC and interpolate bad channels ----
    ransac = Ransac(**ransac_kw)
    _ = ransac.fit_transform(epochs)
    bad_channels = list(getattr(ransac, "bad_chs_", []))

    if verbose:
        print(f"→ RANSAC bad channels: {bad_channels}")
        print("→ Interpolating bad channels on raw...")

    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=reset_bads)

    # Store useful outputs (lightweight, but very handy)
    state["events_ransac"] = events
    state["epochs_ransac"] = epochs
    state["bad_channels"] = bad_channels
    state["raw"] = raw
    return state



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
        "fixed_length_epochs": step_fixed_length_epochs # Create fixed-length events + epochs (final segments)
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

    