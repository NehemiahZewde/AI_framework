# eeg_preprocess.py

from __future__ import annotations

import matplotlib.pyplot as plt
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import mne
from autoreject import Ransac, get_rejection_threshold
from mne.preprocessing import compute_current_source_density
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
import pandas as pd
from scipy.io import loadmat
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
from mne_icalabel.config import ICALABEL_METHODS_NUMERICAL_TO_STRING
from mne_icalabel import label_components
import re

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
    .bdf, .edf, .cnt, .set, .fif, .mff, .egi, .raw, .gdf, .cdt

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
        ".cdt": mne.io.read_raw_curry,
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


def _normalize_annotation_label(value: Any) -> str:
    """Normalize annotation text for robust condition matching."""
    text = str(value).strip().casefold()
    text = re.sub(r"[_-]+", " ", text)   # Treat underscores/hyphens as spaces
    text = re.sub(r"\s+", " ", text)     # Collapse repeated whitespace
    return text


# def step_detect_analysis_conditions(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Detect analysis conditions from Raw annotations and reconstruct their intervals.

#     This step only INTERPRETS annotations. It does not crop, filter, epoch, or
#     otherwise modify the EEG signal.

#     Default recognized conditions
#     -----------------------------
#     EO : Eyes Open
#     EC : Eyes Closed

#     Optional params
#     ---------------
#     condition_aliases : dict
#         Canonical condition names mapped to possible annotation labels.

#         Example:
#             {
#                 "EO": ["Eyes Open", "EO", "EyesOpen"],
#                 "EC": ["Eyes Closed", "EC", "EyesClosed"],
#             }

#     store_key : str
#         State key for reconstructed condition intervals.
#         Default: "analysis_conditions"

#     Returns
#     -------
#     state : dict
#         Adds:
#         - state["analysis_conditions"]
#         - state["analysis_condition_markers"]
#         - state["analysis_condition_summary"]
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     # Default condition vocabulary; users can override/extend this later
#     condition_aliases = params.get("condition_aliases", {
#         "EO": ["Eyes Open", "EO", "EyesOpen"],
#         "EC": ["Eyes Closed", "EC", "EyesClosed"],
#     })
#     if not isinstance(condition_aliases, Mapping):
#         raise TypeError("condition_aliases must be a mapping.")

#     # Build normalized annotation label -> canonical condition lookup
#     alias_lookup: dict[str, str] = {}
#     for condition, aliases in condition_aliases.items():
#         aliases = [aliases] if isinstance(aliases, str) else list(aliases)
#         for label in [condition, *aliases]:
#             normalized = _normalize_annotation_label(label)
#             if normalized in alias_lookup and alias_lookup[normalized] != str(condition):
#                 raise ValueError(f"Annotation alias '{label}' maps to multiple conditions.")
#             alias_lookup[normalized] = str(condition)

#     # Find recognized condition markers without altering other annotations
#     markers: list[dict[str, Any]] = []
#     other_descriptions: list[str] = []
#     for onset, duration, description in zip(
#         raw.annotations.onset,
#         raw.annotations.duration,
#         raw.annotations.description,
#     ):
#         condition = alias_lookup.get(_normalize_annotation_label(description))
#         if condition is None:
#             other_descriptions.append(str(description))
#             continue

#         # Convert annotation onset safely to a sample in the current Raw object
#         sample = int(raw.time_as_index(
#             [float(onset)],
#             use_rounding=True,
#             origin=raw.annotations.orig_time,
#         )[0])

#         if 0 <= sample < raw.n_times:
#             markers.append({
#                 "condition": condition,
#                 "description": str(description),
#                 "sample": sample,
#                 "onset_sec": sample / float(raw.info["sfreq"]),
#                 "annotation_duration_sec": float(duration),
#             })

#     markers.sort(key=lambda x: x["sample"])

#     # Reconstruct intervals from each condition marker to the next condition marker
#     sfreq = float(raw.info["sfreq"])
#     intervals: dict[str, list[dict[str, Any]]] = {}

#     for i, marker in enumerate(markers):
#         start = marker["sample"]
#         next_start = markers[i + 1]["sample"] if i + 1 < len(markers) else raw.n_times

#         # If the annotation itself has a duration, respect it but never cross
#         # into the next recognized condition.
#         duration_sec = marker["annotation_duration_sec"]
#         if duration_sec > 0:
#             duration_stop = start + int(round(duration_sec * sfreq))
#             stop = min(next_start, duration_stop, raw.n_times)
#         else:
#             stop = min(next_start, raw.n_times)

#         if stop <= start:
#             continue

#         condition = marker["condition"]
#         intervals.setdefault(condition, []).append({
#             "start_sample": int(start),
#             "stop_sample": int(stop),
#             "start_sec": float(start / sfreq),
#             "stop_sec": float(stop / sfreq),
#             "duration_sec": float((stop - start) / sfreq),
#         })

#     condition_names = list(dict.fromkeys(marker["condition"] for marker in markers))
#     other_descriptions = list(dict.fromkeys(other_descriptions))

#     # Store interpretation results for later preprocessing steps
#     store_key = str(params.get("store_key", "analysis_conditions"))
#     state[store_key] = intervals
#     state["analysis_condition_markers"] = markers
#     state["analysis_condition_summary"] = {
#         "conditions_detected": bool(intervals),
#         "condition_names": condition_names,
#         "n_condition_markers": len(markers),
#         "n_blocks_by_condition": {
#             condition: len(intervals.get(condition, []))
#             for condition in condition_names
#         },
#         "other_annotation_descriptions": other_descriptions,
#     }

#     if verbose:
#         print("\nAnalysis-condition detection")
#         print("-" * 40)
#         if not intervals:
#             print("No recognized analysis conditions found.")
#         else:
#             print(f"Detected conditions: {', '.join(condition_names)}")
#             print(f"Condition markers: {len(markers)}")
#             for condition in condition_names:
#                 blocks = intervals.get(condition, [])
#                 total_sec = sum(block["duration_sec"] for block in blocks)
#                 print(f"{condition}: {len(blocks)} block(s), {total_sec / 60:.2f} min")
#             if other_descriptions:
#                 print(
#                     "Other annotations retained but not used as conditions: "
#                     + ", ".join(other_descriptions)
#                 )

#     return state



def step_detect_analysis_conditions(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Detect analysis conditions from Raw annotations and reconstruct their intervals.

    This step only INTERPRETS annotations. It does not crop, filter, epoch, or
    otherwise modify the EEG signal.

    The function preserves the general behavior of the original pipeline:

    - No recognized conditions:
        No condition-specific intervals are created.

    - One recognized condition:
        The original marker-to-next-marker behavior is retained.

    - Two recognized conditions:
        The annotation sequence is checked for alternation.

        Example:
            EC -> EO -> EC -> EO

        Consecutive repeated condition markers such as:

            EC -> EC
            EO -> EO

        are treated as ambiguous. The interval between those two markers is
        NOT assigned to either analysis condition.

        Importantly, the function does not invent a missing condition or
        change an annotation label.

    - More than two recognized conditions:
        The original general marker-to-next-marker behavior is retained.

    For a two-condition sequence, the function also calculates the typical
    interval duration from the correctly alternating transitions. This is
    stored for QC/traceability only; no study-specific duration threshold is
    imposed.

    Default recognized conditions
    -----------------------------
    EO : Eyes Open
    EC : Eyes Closed

    Optional params
    ---------------
    condition_aliases : dict
        Canonical condition names mapped to possible annotation labels.

        Example:
            {
                "EO": ["Eyes Open", "EO", "EyesOpen"],
                "EC": ["Eyes Closed", "EC", "EyesClosed"],
            }

    validate_two_condition_alternation : bool
        Whether to validate alternation when exactly two distinct analysis
        conditions are present.

        Default: True.

        This has no effect when zero, one, or more than two distinct
        conditions are detected.

    store_key : str
        State key for reconstructed condition intervals.
        Default: "analysis_conditions"

    Returns
    -------
    state : dict
        Adds:
        - state["analysis_conditions"]
        - state["analysis_condition_markers"]
        - state["analysis_condition_interval_qc_df"]
        - state["analysis_condition_summary"]

    Notes
    -----
    - Non-condition annotations such as Movement or Talking are preserved
      for QC but do not automatically change an analysis condition.
    - Repeated two-condition markers are treated conservatively:
      the ambiguous interval is excluded rather than guessed.
    - The function does not assume a fixed interval duration.
    """

    # ------------------------------------------------------------------
    # 0) Validate Raw input
    # ------------------------------------------------------------------
    raw = state.get("raw")

    if raw is None:
        raise RuntimeError(
            "No raw in state. Did you run 'load_eeg' first?"
        )

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            "state['raw'] must contain an MNE Raw object."
        )

    if params is None:
        params = {}

    if not isinstance(params, Mapping):
        raise TypeError(
            "params must be a mapping."
        )

    # ------------------------------------------------------------------
    # 1) Default condition vocabulary
    # ------------------------------------------------------------------
    # Users can override or extend this for other datasets.
    condition_aliases = params.get(
        "condition_aliases",
        {
            "EO": ["Eyes Open", "EO", "EyesOpen"],
            "EC": ["Eyes Closed", "EC", "EyesClosed"],
        },
    )

    if not isinstance(condition_aliases, Mapping):
        raise TypeError(
            "condition_aliases must be a mapping."
        )

    validate_two_condition_alternation = bool(
        params.get(
            "validate_two_condition_alternation",
            True,
        )
    )

    store_key = str(
        params.get(
            "store_key",
            "analysis_conditions",
        )
    )

    # ------------------------------------------------------------------
    # 2) Build normalized annotation label -> canonical condition lookup
    # ------------------------------------------------------------------
    alias_lookup: dict[str, str] = {}

    for condition, aliases in condition_aliases.items():

        aliases = (
            [aliases]
            if isinstance(aliases, str)
            else list(aliases)
        )

        for label in [condition, *aliases]:

            normalized = _normalize_annotation_label(
                label
            )

            if (
                normalized in alias_lookup
                and alias_lookup[normalized] != str(condition)
            ):
                raise ValueError(
                    f"Annotation alias '{label}' maps to multiple conditions."
                )

            alias_lookup[normalized] = str(
                condition
            )

    # ------------------------------------------------------------------
    # 3) Read the complete annotation timeline
    # ------------------------------------------------------------------
    #
    # Recognized analysis-condition annotations are stored in `markers`.
    # All other annotation descriptions are preserved separately for QC.
    # ------------------------------------------------------------------
    markers: list[dict[str, Any]] = []
    other_annotations: list[dict[str, Any]] = []

    for annotation_idx, (
        onset,
        duration,
        description,
    ) in enumerate(
        zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
    ):

        description = str(
            description
        )

        condition = alias_lookup.get(
            _normalize_annotation_label(
                description
            )
        )

        # Convert annotation onset safely to a sample in the current Raw object.
        sample = int(
            raw.time_as_index(
                [float(onset)],
                use_rounding=True,
                origin=raw.annotations.orig_time,
            )[0]
        )

        if condition is None:

            other_annotations.append({
                "annotation_idx":
                    int(annotation_idx),

                "description":
                    description,

                "sample":
                    int(sample),

                "onset_sec":
                    float(onset),

                "duration_sec":
                    float(duration),
            })

            continue

        if 0 <= sample < raw.n_times:

            markers.append({
                "annotation_idx":
                    int(annotation_idx),

                "condition":
                    condition,

                "description":
                    description,

                "sample":
                    int(sample),

                "onset_sec":
                    sample
                    / float(
                        raw.info["sfreq"]
                    ),

                "annotation_duration_sec":
                    float(duration),
            })

    markers.sort(
        key=lambda x: x["sample"]
    )

    # ------------------------------------------------------------------
    # 4) Determine the detected condition structure
    # ------------------------------------------------------------------
    condition_names = list(
        dict.fromkeys(
            marker["condition"]
            for marker in markers
        )
    )

    n_conditions = len(
        condition_names
    )

    use_alternation_validation = (
        validate_two_condition_alternation
        and n_conditions == 2
        and len(markers) >= 2
    )

    # ------------------------------------------------------------------
    # 5) Reconstruct candidate intervals
    # ------------------------------------------------------------------
    sfreq = float(
        raw.info["sfreq"]
    )

    intervals: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    interval_qc_rows: list[
        dict[str, Any]
    ] = []

    for i, marker in enumerate(
        markers
    ):

        start = int(
            marker["sample"]
        )

        has_next_marker = (
            i + 1
            < len(markers)
        )

        next_marker = (
            markers[i + 1]
            if has_next_marker
            else None
        )

        next_start = (
            int(
                next_marker["sample"]
            )
            if has_next_marker
            else int(
                raw.n_times
            )
        )

        next_condition = (
            str(
                next_marker["condition"]
            )
            if has_next_marker
            else None
        )

        # --------------------------------------------------------------
        # Respect explicit annotation duration when one is available.
        # Otherwise, use the next recognized condition marker.
        # --------------------------------------------------------------
        duration_sec = float(
            marker[
                "annotation_duration_sec"
            ]
        )

        if duration_sec > 0:

            duration_stop = (
                start
                + int(
                    round(
                        duration_sec
                        * sfreq
                    )
                )
            )

            stop = min(
                next_start,
                duration_stop,
                raw.n_times,
            )

        else:

            stop = min(
                next_start,
                raw.n_times,
            )

        if stop <= start:
            continue

        condition = str(
            marker["condition"]
        )

        interval_duration_sec = float(
            (
                stop
                - start
            )
            / sfreq
        )

        # --------------------------------------------------------------
        # Determine whether the interval is structurally valid.
        #
        # The alternation rule is only applied when exactly TWO distinct
        # conditions were detected.
        #
        # Examples:
        #
        #     EC -> EO  = valid
        #     EO -> EC  = valid
        #
        #     EC -> EC  = ambiguous
        #     EO -> EO  = ambiguous
        #
        # The final marker has no following condition marker, so it cannot
        # be tested for alternation and is retained using the original
        # end-of-recording behavior.
        # --------------------------------------------------------------
        if (
            use_alternation_validation
            and has_next_marker
        ):

            alternation_valid = (
                condition
                != next_condition
            )

        else:

            alternation_valid = None

        # --------------------------------------------------------------
        # Preserve any non-condition annotations that occur inside this
        # candidate interval.
        #
        # These are QC information only. They do not automatically change
        # the eye-state assignment.
        # --------------------------------------------------------------
        intervening_annotations = [
            annotation[
                "description"
            ]

            for annotation
            in other_annotations

            if (
                annotation[
                    "sample"
                ] > start
                and annotation[
                    "sample"
                ] < stop
            )
        ]

        # --------------------------------------------------------------
        # Decide whether this interval should be retained.
        # --------------------------------------------------------------
        #
        # For a validated two-condition sequence, repeated consecutive
        # condition markers create an ambiguous interval.
        #
        # We exclude that interval rather than assuming the previous
        # condition continued throughout it.
        # --------------------------------------------------------------
        if (
            use_alternation_validation
            and has_next_marker
            and alternation_valid is False
        ):

            accepted = False

            decision_reason = (
                "repeated_condition_marker"
            )

        else:

            accepted = True

            if (
                use_alternation_validation
                and has_next_marker
            ):

                decision_reason = (
                    "accepted_alternating_interval"
                )

            elif (
                use_alternation_validation
                and not has_next_marker
            ):

                decision_reason = (
                    "accepted_terminal_interval"
                )

            else:

                decision_reason = (
                    "accepted_original_behavior"
                )

        # --------------------------------------------------------------
        # Store QC for EVERY candidate interval, including rejected ones.
        # --------------------------------------------------------------
        interval_qc_rows.append({
            "condition":
                condition,

            "next_condition":
                next_condition,

            "start_sample":
                int(start),

            "stop_sample":
                int(stop),

            "start_sec":
                float(
                    start
                    / sfreq
                ),

            "stop_sec":
                float(
                    stop
                    / sfreq
                ),

            "duration_sec":
                interval_duration_sec,

            "has_next_condition_marker":
                bool(
                    has_next_marker
                ),

            "alternation_valid":
                alternation_valid,

            "accepted":
                bool(
                    accepted
                ),

            "decision_reason":
                decision_reason,

            "intervening_annotations":
                intervening_annotations,

            "n_intervening_annotations":
                len(
                    intervening_annotations
                ),
        })

        # --------------------------------------------------------------
        # Only accepted intervals enter downstream analysis.
        # --------------------------------------------------------------
        if not accepted:
            continue

        intervals.setdefault(
            condition,
            [],
        ).append({
            "start_sample":
                int(start),

            "stop_sample":
                int(stop),

            "start_sec":
                float(
                    start
                    / sfreq
                ),

            "stop_sec":
                float(
                    stop
                    / sfreq
                ),

            "duration_sec":
                interval_duration_sec,
        })

    # ------------------------------------------------------------------
    # 6) Build interval-level QC table
    # ------------------------------------------------------------------
    interval_qc_df = pd.DataFrame(
        interval_qc_rows
    )

    state[
        "analysis_condition_interval_qc_df"
    ] = interval_qc_df

    # ------------------------------------------------------------------
    # 7) Learn the typical timing from VALID alternating transitions
    # ------------------------------------------------------------------
    #
    # This is descriptive QC only.
    #
    # We do NOT specify that intervals must be 60 seconds, 90 seconds,
    # or any other study-specific duration.
    #
    # Instead, when a two-condition alternating sequence is present,
    # we summarize the durations of intervals that actually alternate.
    # ------------------------------------------------------------------
    typical_alternating_interval_sec = None
    alternating_interval_mad_sec = None
    n_valid_alternating_intervals = 0

    if (
        use_alternation_validation
        and not interval_qc_df.empty
    ):

        valid_alternating_df = (
            interval_qc_df.loc[
                (
                    interval_qc_df[
                        "alternation_valid"
                    ] == True
                )
                &
                (
                    interval_qc_df[
                        "accepted"
                    ] == True
                )
            ]
        )

        n_valid_alternating_intervals = int(
            len(
                valid_alternating_df
            )
        )

        if (
            n_valid_alternating_intervals
            > 0
        ):

            valid_durations = (
                valid_alternating_df[
                    "duration_sec"
                ]
                .astype(float)
                .to_numpy()
            )

            typical_alternating_interval_sec = float(
                np.median(
                    valid_durations
                )
            )

            alternating_interval_mad_sec = float(
                np.median(
                    np.abs(
                        valid_durations
                        - typical_alternating_interval_sec
                    )
                )
            )

    # ------------------------------------------------------------------
    # 8) Determine whether the two-condition sequence remains analyzable
    # ------------------------------------------------------------------
    #
    # If we detected a two-condition alternating design but, after removing
    # ambiguous repeated-marker intervals, one condition has no usable
    # intervals left, the recording cannot be interpreted safely.
    #
    # This should stop only THIS recording in the batch workflow.
    # ------------------------------------------------------------------
    annotation_status = (
        "not_applicable"
    )

    annotation_error = None

    if use_alternation_validation:

        annotation_status = (
            "ok"
        )

        conditions_without_intervals = [
            condition

            for condition
            in condition_names

            if not intervals.get(
                condition,
                []
            )
        ]

        if conditions_without_intervals:

            annotation_status = (
                "cannot_calculate"
            )

            annotation_error = (
                "Alternating condition annotations could not be "
                "resolved safely. No accepted intervals remain for: "
                + ", ".join(
                    conditions_without_intervals
                )
            )

    # ------------------------------------------------------------------
    # 9) Preserve non-condition annotation descriptions
    # ------------------------------------------------------------------
    other_descriptions = list(
        dict.fromkeys(
            annotation[
                "description"
            ]
            for annotation
            in other_annotations
        )
    )

    # ------------------------------------------------------------------
    # 10) Store interpretation results for later preprocessing steps
    # ------------------------------------------------------------------
    state[
        store_key
    ] = intervals

    state[
        "analysis_condition_markers"
    ] = markers

    state[
        "analysis_condition_summary"
    ] = {

        "conditions_detected":
            bool(
                intervals
            ),

        "condition_names":
            condition_names,

        "n_conditions_detected":
            int(
                n_conditions
            ),

        "n_condition_markers":
            len(
                markers
            ),

        "n_blocks_by_condition": {
            condition:
                len(
                    intervals.get(
                        condition,
                        [],
                    )
                )

            for condition
            in condition_names
        },

        "duration_seconds_by_condition": {
            condition:
                float(
                    sum(
                        block[
                            "duration_sec"
                        ]

                        for block
                        in intervals.get(
                            condition,
                            []
                        )
                    )
                )

            for condition
            in condition_names
        },

        "other_annotation_descriptions":
            other_descriptions,

        # Alternation QC
        "alternation_validation_applied":
            bool(
                use_alternation_validation
            ),

        "annotation_status":
            annotation_status,

        "annotation_error":
            annotation_error,

        "n_valid_alternating_intervals":
            int(
                n_valid_alternating_intervals
            ),

        # Learned from this recording itself.
        "typical_alternating_interval_sec":
            typical_alternating_interval_sec,

        "alternating_interval_mad_sec":
            alternating_interval_mad_sec,

        "n_ambiguous_intervals":
            int(
                (
                    interval_qc_df[
                        "accepted"
                    ] == False
                ).sum()
            )
            if not interval_qc_df.empty
            else 0,
    }

    # ------------------------------------------------------------------
    # 11) If the alternating sequence cannot be calculated safely,
    #     stop THIS physical recording.
    #
    # build_label_epoch_arrays() catches recording-level preprocessing
    # failures, records the reason in QC, and continues the cohort.
    # ------------------------------------------------------------------
    if (
        annotation_status
        == "cannot_calculate"
    ):

        raise RuntimeError(
            "Analysis conditions cannot be calculated. "
            + str(
                annotation_error
            )
        )

    # ------------------------------------------------------------------
    # 12) Optional reporting
    # ------------------------------------------------------------------
    if verbose:

        print(
            "\nAnalysis-condition detection"
        )

        print(
            "-" * 40
        )

        if not intervals:

            print(
                "No recognized analysis conditions found."
            )

        else:

            print(
                "Detected conditions: "
                + ", ".join(
                    condition_names
                )
            )

            print(
                f"Condition markers: "
                f"{len(markers)}"
            )

            for condition in (
                condition_names
            ):

                blocks = intervals.get(
                    condition,
                    []
                )

                total_sec = sum(
                    block[
                        "duration_sec"
                    ]
                    for block
                    in blocks
                )

                print(
                    f"{condition}: "
                    f"{len(blocks)} block(s), "
                    f"{total_sec / 60:.2f} min"
                )

            # ----------------------------------------------------------
            # Show alternation-specific QC only when it was applicable.
            # ----------------------------------------------------------
            if use_alternation_validation:

                print(
                    "Alternation validation: "
                    f"{annotation_status.upper()}"
                )

                print(
                    "Valid alternating intervals: "
                    f"{n_valid_alternating_intervals}"
                )

                n_ambiguous = int(
                    (
                        interval_qc_df[
                            "accepted"
                        ] == False
                    ).sum()
                )

                print(
                    "Ambiguous intervals excluded: "
                    f"{n_ambiguous}"
                )

                if (
                    typical_alternating_interval_sec
                    is not None
                ):

                    print(
                        "Typical alternating interval "
                        "(learned from annotations): "
                        f"{typical_alternating_interval_sec:.2f} sec"
                    )

            if other_descriptions:

                print(
                    "Other annotations retained but not used as conditions: "
                    + ", ".join(
                        other_descriptions
                    )
                )

    return state


def step_mark_non_analysis_segments(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Mark portions outside recognized analysis conditions as BAD_non_analysis."""
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    conditions_key = str(params.get("conditions_key", "analysis_conditions"))
    description = str(params.get("description", "BAD_non_analysis"))
    conditions = state.get(conditions_key, {})

    # No recognized conditions -> preserve normal single-recording behavior
    if not conditions:
        state["non_analysis_summary"] = {
            "applied": False,
            "reason": "no_analysis_conditions_detected",
            "n_segments": 0,
            "total_duration_sec": 0.0,
        }
        if verbose:
            print("→ No analysis conditions detected; no non-analysis segments marked.")
        return state

    # Collect and merge all recognized analysis intervals
    n_times = int(raw.n_times)
    sfreq = float(raw.info["sfreq"])
    intervals = []
    for blocks in conditions.values():
        for block in blocks:
            start = max(0, int(block["start_sample"]))
            stop = min(n_times, int(block["stop_sample"]))
            if stop > start:
                intervals.append((start, stop))

    intervals.sort()
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))

    # Find the complement: portions NOT belonging to an analysis condition
    non_analysis = []
    cursor = 0
    for start, stop in merged:
        if start > cursor:
            non_analysis.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor < n_times:
        non_analysis.append((cursor, n_times))

    # Make step safe to rerun by removing its previous annotations
    old_idx = [
        i for i, desc in enumerate(raw.annotations.description)
        if str(desc) == description
    ]
    if old_idx:
        raw.annotations.delete(old_idx)

    # Add BAD annotations while preserving the original recording timeline
    first_time = float(raw.first_time) if raw.annotations.orig_time is not None else 0.0
    for start, stop in non_analysis:
        raw.annotations.append(
            onset=first_time + start / sfreq,
            duration=(stop - start) / sfreq,
            description=description,
        )

    total_sec = sum((stop - start) / sfreq for start, stop in non_analysis)
    state["raw"] = raw
    state["non_analysis_segments"] = [
        {
            "start_sample": start,
            "stop_sample": stop,
            "start_sec": start / sfreq,
            "stop_sec": stop / sfreq,
            "duration_sec": (stop - start) / sfreq,
        }
        for start, stop in non_analysis
    ]
    state["non_analysis_summary"] = {
        "applied": True,
        "description": description,
        "n_segments": len(non_analysis),
        "total_duration_sec": float(total_sec),
    }

    if verbose:
        print("\nNon-analysis marking")
        print("-" * 40)
        print(f"Segments marked: {len(non_analysis)}")
        print(f"Total excluded time: {total_sec:.2f} s")
        for start, stop in non_analysis:
            print(f"BAD_non_analysis: {start / sfreq:.2f}–{stop / sfreq:.2f} s")

    return state


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


def _compute_eeg_psd_scale_metric(
    raw: mne.io.BaseRaw,
    *,
    psd_params: Dict[str, Any],
    metric: str = "median",
) -> float:
    """
    Compute one whole-recording PSD magnitude metric for EEG scale checking.

    The PSD calculation itself is controlled entirely by psd_params, which
    are passed directly to raw.compute_psd().

    PSD values returned by MNE are based on SI units. For EEG, convert
    V²/Hz -> µV²/Hz before expressing the PSD in dB so the numerical
    values correspond to the familiar EEG PSD plotting scale.

    Parameters
    ----------
    raw
        Loaded MNE Raw object.

    psd_params
        Keyword arguments passed directly to raw.compute_psd().

    metric
        Summary statistic applied across all returned PSD values.

        Supported:
            "median"
            "mean"

    Returns
    -------
    float
        Whole-recording PSD scale metric in dB re 1 µV²/Hz.
    """

    if not isinstance(psd_params, dict):
        raise TypeError(
            "psd_params must be a dictionary."
        )

    # ------------------------------------------------------------
    # Compute PSD using the parameters supplied by the config.
    # ------------------------------------------------------------
    spectrum = raw.compute_psd(
        **psd_params
    )

    psd_v2_per_hz = np.asarray(
        spectrum.get_data(),
        dtype=float,
    )

    if psd_v2_per_hz.size == 0:
        raise RuntimeError(
            "PSD calculation returned no values."
        )

    # ------------------------------------------------------------
    # Convert V²/Hz -> µV²/Hz.
    #
    # 1 V = 1e6 µV
    # therefore:
    # 1 V² = 1e12 µV²
    # ------------------------------------------------------------
    psd_uv2_per_hz = (
        psd_v2_per_hz * 1e12
    )

    tiny = np.finfo(float).tiny

    psd_db = (
        10.0
        * np.log10(
            np.maximum(
                psd_uv2_per_hz,
                tiny,
            )
        )
    )

    finite_values = psd_db[
        np.isfinite(psd_db)
    ]

    if finite_values.size == 0:
        raise RuntimeError(
            "PSD calculation produced no finite values."
        )

    # ------------------------------------------------------------
    # User-configurable summary metric.
    # ------------------------------------------------------------
    metric = str(metric).lower().strip()

    if metric == "median":
        value = np.median(
            finite_values
        )

    elif metric == "mean":
        value = np.mean(
            finite_values
        )

    else:
        raise ValueError(
            "metric must be 'median' or 'mean'."
        )

    return float(value)


def step_auto_scale_data(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Automatically detect and correct a large EEG unit-scale mismatch.

    Workflow
    --------
    1. Compute a whole-recording EEG PSD.
    2. Reduce the PSD to one configurable summary metric.
    3. Check whether that metric is within the configured physiological range.
    4. If it is clearly far outside the range, calculate the required
       power-of-ten amplitude correction.
    5. Apply the correction.
    6. Recompute the PSD and verify that the corrected metric falls inside
       the physiological range.

    Important
    ---------
    This is intended to correct UNIT SCALE, not normalize biological EEG
    amplitude between subjects.

    Required params
    ---------------
    psd : dict
        Passed directly to raw.compute_psd().

    physiological_psd_db_range : tuple[float, float]
        Acceptable range for the PSD summary metric.

    Optional params
    ---------------
    metric : str
        "median" or "mean". Default: "median".

    auto_scale_trigger_margin_db : float
        How far outside the physiological range the metric must be before
        automatic scaling is allowed. Default: 40 dB.

    scale_exponent_step : int
        Allowed spacing between scaling exponents.

        1 -> any power of ten:
             ..., 1e-8, 1e-7, 1e-6, 1e-5, ...

        3 -> SI-style thousand-fold steps:
             ..., 1e-9, 1e-6, 1e-3, 1, 1e3, ...

        Default: 3.

    apply : dict
        Parameters passed directly to raw.apply_function().
        Example:
            {
                "picks": "eeg",
                "channel_wise": False,
                "verbose": False,
            }
    """

    raw = state.get("raw")

    if raw is None:
        raise RuntimeError(
            "No raw in state. Did you run 'load_eeg' first?"
        )

    raw.load_data()

    # ============================================================
    # Read configuration
    # ============================================================
    psd_params = params.get(
        "psd",
        None,
    )

    if not isinstance(psd_params, dict):
        raise TypeError(
            "auto_scale_data requires params['psd'] as a dictionary."
        )

    physiological_range = params.get(
        "physiological_psd_db_range",
        None,
    )

    if (
        not isinstance(physiological_range, (tuple, list))
        or len(physiological_range) != 2
    ):
        raise ValueError(
            "physiological_psd_db_range must contain "
            "(lower_db, upper_db)."
        )

    lower_db = float(
        physiological_range[0]
    )

    upper_db = float(
        physiological_range[1]
    )

    if upper_db <= lower_db:
        raise ValueError(
            "physiological_psd_db_range must satisfy "
            "lower < upper."
        )

    metric = str(
        params.get(
            "metric",
            "median",
        )
    )

    trigger_margin_db = float(
        params.get(
            "auto_scale_trigger_margin_db",
            40.0,
        )
    )

    if trigger_margin_db < 0:
        raise ValueError(
            "auto_scale_trigger_margin_db must be >= 0."
        )

    exponent_step = int(
        params.get(
            "scale_exponent_step",
            3,
        )
    )

    if exponent_step < 1:
        raise ValueError(
            "scale_exponent_step must be >= 1."
        )

    apply_params = params.get(
        "apply",
        {
            "picks": "eeg",
            "channel_wise": False,
            "verbose": False,
        },
    )

    if not isinstance(apply_params, dict):
        raise TypeError(
            "params['apply'] must be a dictionary."
        )

    # ============================================================
    # 1. PSD BEFORE scaling
    # ============================================================
    psd_before_db = (
        _compute_eeg_psd_scale_metric(
            raw,
            psd_params=psd_params,
            metric=metric,
        )
    )

    # ============================================================
    # 2. Already within physiological range?
    # ============================================================
    within_range_before = (
        lower_db
        <= psd_before_db
        <= upper_db
    )

    if within_range_before:

        scale_exponent = 0
        scale_factor = 1.0
        psd_after_db = psd_before_db

        state["raw"] = raw
        state["scale_factor"] = scale_factor

        state["auto_scale_summary"] = {
            "metric": metric,
            "physiological_psd_db_range": (
                lower_db,
                upper_db,
            ),
            "psd_before_db": psd_before_db,
            "psd_after_db": psd_after_db,
            "scale_exponent": scale_exponent,
            "scale_factor": scale_factor,
            "scaling_applied": False,
            "status": "already_within_range",
        }

        if verbose:
            print("\nAutomatic EEG scaling")
            print("-" * 50)
            print(
                f"PSD {metric}: "
                f"{psd_before_db:.2f} dB"
            )
            print(
                "Scale already within configured "
                "physiological range."
            )
            print("Scale factor: 1")

        return state

    # ============================================================
    # 3. How far outside the range are we?
    # ============================================================
    if psd_before_db < lower_db:
        distance_outside_db = (
            lower_db - psd_before_db
        )
    else:
        distance_outside_db = (
            psd_before_db - upper_db
        )

    # ------------------------------------------------------------
    # Slightly outside the range:
    #
    # DO NOT automatically alter the data.
    #
    # This protects against genuine biological variation.
    # ------------------------------------------------------------
    if distance_outside_db < trigger_margin_db:
        raise RuntimeError(
            "EEG PSD scale is outside the configured physiological "
            "range, but not far enough outside it to safely infer a "
            "unit-scale mismatch.\n"
            f"PSD {metric}: {psd_before_db:.2f} dB\n"
            f"Configured range: {lower_db:.2f} to {upper_db:.2f} dB\n"
            f"Distance outside range: {distance_outside_db:.2f} dB\n"
            "No scaling was applied."
        )

    # ============================================================
    # 4. Calculate required scale exponent
    # ============================================================

    # Midpoint is used only to determine which UNIT SCALE
    # is most consistent with the configured physiological range.
    #
    # We still apply only a discrete power-of-ten correction.
    midpoint_db = (
        lower_db + upper_db
    ) / 2.0

    # ------------------------------------------------------------
    # If signal amplitude is multiplied by 10^k:
    #
    # PSD shifts by 20*k dB.
    #
    # Therefore:
    #
    # k = desired dB shift / 20
    # ------------------------------------------------------------
    estimated_exponent = (
        midpoint_db - psd_before_db
    ) / 20.0

    # Snap to the configured exponent spacing.
    #
    # Example with exponent_step=3:
    # -5.89 -> -6
    scale_exponent = int(
        np.rint(
            estimated_exponent
            / exponent_step
        )
        * exponent_step
    )

    scale_factor = float(
        10.0 ** scale_exponent
    )

    # ------------------------------------------------------------
    # Predict where the PSD should land BEFORE touching the data.
    # ------------------------------------------------------------
    predicted_psd_db = (
        psd_before_db
        + 20.0 * scale_exponent
    )

    predicted_within_range = (
        lower_db
        <= predicted_psd_db
        <= upper_db
    )

    if not predicted_within_range:
        raise RuntimeError(
            "Automatic scaling could not identify a safe "
            "power-of-ten correction.\n"
            f"PSD before: {psd_before_db:.2f} dB\n"
            f"Estimated exponent: {estimated_exponent:.3f}\n"
            f"Selected exponent: {scale_exponent}\n"
            f"Predicted PSD after scaling: "
            f"{predicted_psd_db:.2f} dB\n"
            "No scaling was applied."
        )

    # ============================================================
    # 5. Apply scaling
    # ============================================================
    raw.apply_function(
        lambda data: data * scale_factor,
        **apply_params,
    )

    # ============================================================
    # 6. Verify with a fresh PSD
    # ============================================================
    psd_after_db = (
        _compute_eeg_psd_scale_metric(
            raw,
            psd_params=psd_params,
            metric=metric,
        )
    )

    verified = (
        lower_db
        <= psd_after_db
        <= upper_db
    )

    if not verified:

        # Undo the scaling before raising the error.
        raw.apply_function(
            lambda data: data / scale_factor,
            **apply_params,
        )

        raise RuntimeError(
            "Automatic EEG scaling failed verification.\n"
            f"PSD before: {psd_before_db:.2f} dB\n"
            f"Scale factor attempted: {scale_factor:g}\n"
            f"PSD after: {psd_after_db:.2f} dB\n"
            f"Required range: {lower_db:.2f} to "
            f"{upper_db:.2f} dB\n"
            "Scaling was reverted."
        )

    # ============================================================
    # 7. Store audit information
    # ============================================================
    state["raw"] = raw
    state["scale_factor"] = scale_factor

    state["auto_scale_summary"] = {
        "metric": metric,
        "physiological_psd_db_range": (
            lower_db,
            upper_db,
        ),
        "psd_before_db": psd_before_db,
        "psd_after_db": psd_after_db,
        "distance_outside_range_db":
            distance_outside_db,
        "estimated_scale_exponent":
            float(estimated_exponent),
        "scale_exponent":
            int(scale_exponent),
        "scale_factor":
            float(scale_factor),
        "scaling_applied":
            True,
        "status":
            "scaled_and_verified",
    }

    if verbose:
        print("\nAutomatic EEG scaling")
        print("-" * 50)

        print(
            f"PSD {metric} before: "
            f"{psd_before_db:.2f} dB"
        )

        print(
            f"Physiological range: "
            f"{lower_db:.2f} to {upper_db:.2f} dB"
        )

        print(
            f"Estimated exponent: "
            f"{estimated_exponent:.3f}"
        )

        print(
            f"Applied exponent: "
            f"{scale_exponent}"
        )

        print(
            f"Scale factor: "
            f"{scale_factor:g}"
        )

        print(
            f"PSD {metric} after: "
            f"{psd_after_db:.2f} dB"
        )

        print(
            "Scaling verified."
        )

    return state


def _safe_high_cut_for_iclabel(sfreq: float, desired: float = 100.0) -> float:
    """
    Keep ICA/ICLabel high cutoff below Nyquist.
    """
    nyq = sfreq / 2.0
    return float(min(desired, nyq - 1.0)) if nyq > 2 else float(nyq * 0.8)





# def step_run_ica_iclabel(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Run ICA on a dedicated filtered copy of raw, classify ICs with ICLabel,
#     exclude selected artifact ICs, and apply the ICA solution back to the
#     original raw signal.

#     This step is suitable for resting-state EEG or ERP EEG, as long as it is
#     applied before CSD / surface Laplacian transformation.

#     Expected params
#     ---------------
#     ica : dict, optional
#         Passed to mne.preprocessing.ICA(...)

#         Example:
#             {
#                 "n_components": 0.99,
#                 "method": "infomax",
#                 "fit_params": {"extended": True},
#                 "random_state": 42,
#                 "max_iter": "auto"
#             }

#     fit : dict, optional
#         Controls preprocessing of the ICA-fit branch.

#         Example:
#             {
#                 "notch_freqs": None,
#                 "l_freq": 1.0,
#                 "desired_h_freq": 100.0,
#                 "apply_average_ref": True,
#                 "picks": "eeg"
#             }

#         Notes:
#             - ICLabel was designed for ICA solutions fitted on EEG that is
#               common-average referenced and filtered roughly between 1 and
#               100 Hz.
#             - The ICA solution is fit on this temporary filtered branch but
#               applied back to the original raw object.

#     iclabel : dict, optional
#         Controls ICLabel-based exclusion.

#         Example:
#             {
#                 "artifact_labels": [
#                     "eye blink",
#                     "muscle artifact",
#                     "line noise",
#                     "heart beat",
#                     "channel noise"
#                 ],
#                 "prob_threshold": 0.8
#             }

#         Labels usually returned by ICLabel:
#             - "brain"
#             - "muscle artifact"
#             - "eye blink"
#             - "heart beat"
#             - "line noise"
#             - "channel noise"
#             - "other"

#     store : dict, optional
#         Keys to store outputs in state.

#         Defaults:
#             {
#                 "raw_key": "raw",
#                 "ica_key": "ica",
#                 "ic_df_key": "iclabel_df",
#                 "exclude_key": "excluded_ics",
#                 "labels_key": "ic_labels",
#                 "proba_key": "ic_probs"
#             }

#     Returns
#     -------
#     state : dict
#         Updated pipeline state containing:
#         - cleaned raw object
#         - fitted ICA object
#         - ICLabel summary DataFrame
#         - excluded component indices
#         - component labels
#         - predicted-label probabilities

#     Notes
#     -----
#     - This step removes artifact *components*, not noisy epochs.
#     - ICA is fit on a filtered copy, then applied to the original raw.
#     - For resting EEG, a cautious probability threshold such as 0.8 or 0.9 is
#       recommended before trusting automatic component rejection.
#     - Requires mne_icalabel to be installed and importable.
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     ica_kw = params.get("ica", {})
#     fit_kw = params.get("fit", {})
#     iclabel_kw = params.get("iclabel", {})
#     store_kw = params.get("store", {})

#     if (
#         not isinstance(ica_kw, dict)
#         or not isinstance(fit_kw, dict)
#         or not isinstance(iclabel_kw, dict)
#         or not isinstance(store_kw, dict)
#     ):
#         raise TypeError(
#             "step_run_ica_iclabel expects dicts for "
#             "'ica', 'fit', 'iclabel', and 'store'."
#         )

#     # -----------------------------
#     # Defaults
#     # -----------------------------
#     # Use extended Infomax by default because ICLabel was designed around this
#     # kind of ICA solution. n_components=0.99 is often safer than None after
#     # interpolation/re-referencing because the effective rank may be reduced.
#     ica_defaults: Dict[str, Any] = {
#         "n_components": 0.99,
#         "method": "infomax",
#         "fit_params": {"extended": True},
#         "random_state": 42,
#         "max_iter": "auto",
#     }

#     # The ICA branch is a temporary copy used only to fit ICA.
#     # The fitted ICA solution is applied back to the original raw object later.
#     fit_defaults: Dict[str, Any] = {
#         "notch_freqs": None,
#         "l_freq": 1.0,
#         "desired_h_freq": 100.0,
#         "apply_average_ref": True,
#         "picks": "eeg",
#         "skip_by_annotation": ("edge", "bad_acq_skip", "BAD_non_analysis"),
#         "reject_by_annotation": True,
#     }

#     # For resting EEG, avoid blindly excluding every non-brain component.
#     # Use a probability threshold so uncertain ICLabel decisions are not
#     # automatically removed.
#     iclabel_defaults: Dict[str, Any] = {
#         "artifact_labels": [
#             "eye blink",
#             "muscle artifact",
#             "line noise",
#             "heart beat",
#             "channel noise",
#         ],
#         "prob_threshold": 0.8,
#     }

#     ica_cfg: Dict[str, Any] = {**ica_defaults, **ica_kw}
#     fit_cfg: Dict[str, Any] = {**fit_defaults, **fit_kw}
#     iclabel_cfg: Dict[str, Any] = {**iclabel_defaults, **iclabel_kw}

#     raw_key: str = str(store_kw.get("raw_key", "raw"))
#     ica_key: str = str(store_kw.get("ica_key", "ica"))
#     ic_df_key: str = str(store_kw.get("ic_df_key", "iclabel_df"))
#     exclude_key: str = str(store_kw.get("exclude_key", "excluded_ics"))
#     labels_key: str = str(store_kw.get("labels_key", "ic_labels"))
#     proba_key: str = str(store_kw.get("proba_key", "ic_probs"))

#     # -----------------------------
#     # ICA fit branch
#     # -----------------------------
#     raw_ic = raw.copy().load_data()
#     picks = fit_cfg.get("picks", "eeg")
#     skip_by_annotation = fit_cfg.get(
#         "skip_by_annotation",
#         ("edge", "bad_acq_skip", "BAD_non_analysis"),
#     )

#     # Optional notch filtering on the ICA-fit copy.
#     # If the main pipeline already notch-filtered the data, this can stay None.
#     notch_freqs = fit_cfg.get("notch_freqs", None)
#     if notch_freqs is not None:
#         if isinstance(notch_freqs, (int, float)):
#             notch_freqs_list = [float(notch_freqs)]
#         else:
#             notch_freqs_list = [float(f) for f in notch_freqs]

#         if len(notch_freqs_list) > 0:
#             if verbose:
#                 print(f"→ ICA branch notch filter: {notch_freqs_list}")
#             raw_ic.notch_filter(
#                 freqs=notch_freqs_list,
#                 picks=picks,
#                 skip_by_annotation=skip_by_annotation,
#                 verbose=False,
#             )


#     # ICLabel expects ICA to be fit on approximately 1-100 Hz data.
#     # This helper keeps the requested high cutoff safely below Nyquist.
#     l_freq = float(fit_cfg.get("l_freq", 1.0))
#     desired_h_freq = float(fit_cfg.get("desired_h_freq", 100.0))
#     h_freq = _safe_high_cut_for_iclabel(
#         sfreq=float(raw_ic.info["sfreq"]),
#         desired=desired_h_freq,
#     )

#     if verbose:
#         print(f"→ ICA branch bandpass: {l_freq}–{h_freq} Hz")

#     raw_ic.filter(
#         l_freq=l_freq,
#         h_freq=h_freq,
#         picks=picks,
#         skip_by_annotation=skip_by_annotation,
#         verbose=False,
#     )



#     # ICLabel was designed for common-average-referenced EEG.
#     if bool(fit_cfg.get("apply_average_ref", True)):
#         if verbose:
#             print("→ ICA branch average reference")
#         raw_ic.set_eeg_reference("average", verbose=False)

#     # -----------------------------
#     # Fit ICA
#     # -----------------------------
#     if verbose:
#         print(f"→ Fitting ICA with params: {ica_cfg}")

#     ica = ICA(
#         n_components=ica_cfg["n_components"],
#         method=ica_cfg["method"],
#         fit_params=ica_cfg["fit_params"],
#         random_state=ica_cfg["random_state"],
#         max_iter=ica_cfg["max_iter"],
#     )

#     ica.fit(
#         raw_ic,
#         picks=picks,
#         reject_by_annotation=bool(fit_cfg.get("reject_by_annotation", True)),
#         verbose=False,
#     )


#     # -----------------------------
#     # ICLabel classification
#     # -----------------------------
#     # Use the public mne_icalabel API. This is safer than calling the lower-level
#     # iclabel_label_components function directly and manually mapping class IDs.
#     ic_labels: Dict[str, Any] = label_components(
#         raw_ic,
#         ica,
#         method="iclabel",
#     )

#     labels = list(ic_labels["labels"])
#     y_pred_proba = np.asarray(ic_labels["y_pred_proba"], dtype=float)

#     if len(labels) != len(y_pred_proba):
#         raise RuntimeError(
#             "ICLabel returned mismatched labels and probabilities: "
#             f"len(labels)={len(labels)}, len(y_pred_proba)={len(y_pred_proba)}"
#         )

#     artifact_labels = {
#         str(label).lower()
#         for label in iclabel_cfg.get("artifact_labels", [])
#     }

#     prob_threshold = iclabel_cfg.get("prob_threshold", 0.8)
#     if prob_threshold is not None:
#         prob_threshold = float(prob_threshold)

#     exclude: list[int] = []
#     for idx, (label, proba) in enumerate(zip(labels, y_pred_proba)):
#         label_lower = str(label).lower()

#         # Only remove components that are both:
#         #   1) labeled as one of the requested artifact classes, and
#         #   2) above the requested confidence threshold.
#         if label_lower in artifact_labels:
#             if prob_threshold is None or float(proba) >= prob_threshold:
#                 exclude.append(idx)

#     if verbose:
#         print("→ ICLabel component summary:")
#         for idx, (label, proba) in enumerate(zip(labels, y_pred_proba)):
#             marker = "EXCLUDE" if idx in exclude else "keep"
#             print(f"   IC {idx:03d}: {label:16s} p={float(proba):.3f} [{marker}]")
#         print(f"→ Excluding ICs: {exclude}")

#     # -----------------------------
#     # Apply ICA back to original raw
#     # -----------------------------
#     # Important:
#     # ICA was fit on raw_ic, the filtered ICA branch. The cleaning is applied
#     # to a copy of the original raw object currently in the pipeline.
#     ica.exclude = exclude
#     raw_clean = raw.copy()
#     ica.apply(raw_clean, exclude=exclude, verbose=False)

#     # -----------------------------
#     # Build IC summary table
#     # -----------------------------
#     ic_df = pd.DataFrame({
#         "ic": np.arange(len(labels), dtype=int),
#         "label": labels,
#         "y_pred_proba": y_pred_proba.astype(float),
#         "excluded": [idx in exclude for idx in range(len(labels))],
#     })

#     # -----------------------------
#     # Store outputs
#     # -----------------------------
#     state[raw_key] = raw_clean
#     state[ica_key] = ica
#     state[ic_df_key] = ic_df
#     state[exclude_key] = exclude
#     state[labels_key] = labels
#     state[proba_key] = y_pred_proba.astype(float).tolist()

#     return state


def step_run_ica_iclabel(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run ICA on a dedicated filtered EEG copy, classify components with ICLabel,
    optionally evaluate ICA components against recorded EOG channels, exclude
    selected artifact components, and apply the ICA solution back to the
    original raw signal.

    This step is suitable for resting-state EEG or ERP EEG, as long as it is
    applied before CSD / surface Laplacian transformation.

    Expected params
    ---------------
    ica : dict, optional
        Passed to mne.preprocessing.ICA(...).

        Example:
            {
                "n_components": 0.99,
                "method": "infomax",
                "fit_params": {"extended": True},
                "random_state": 42,
                "max_iter": "auto"
            }

    fit : dict, optional
        Controls preprocessing of the ICA-fit branch.

        Example:
            {
                "notch_freqs": None,
                "l_freq": 1.0,
                "desired_h_freq": 45.0,
                "apply_average_ref": False,
                "picks": "eeg"
            }

        Notes:
            - ICA is fit using EEG channels only.
            - The ICA-fit branch can use a different frequency range from the
              final qEEG signal.
            - The NeuShen configuration currently uses 1-45 Hz.
            - The fitted ICA solution is applied back to the original raw
              object after component selection.

    iclabel : dict, optional
        Controls ICLabel-based component exclusion.

        Example:
            {
                "artifact_labels": [
                    "eye blink",
                    "muscle artifact",
                    "line noise",
                    "heart beat",
                    "channel noise"
                ],
                "prob_threshold": 0.8
            }

    eog : dict, optional
        Controls EOG-supported ocular-component QC.

        Example:
            {
                "enabled": True,
                "ch_names": None,
                "threshold": 3.0,
                "measure": "zscore",
                "l_freq": 1.0,
                "h_freq": 10.0,
                "reject_by_annotation": True,
                "use_for_exclusion": False
            }

        If ch_names is None, all channels explicitly typed as EOG are used.

        IMPORTANT:
            use_for_exclusion=False means EOG information is diagnostic only.
            ICLabel remains responsible for automatic component exclusion.
            This is the recommended setting for the initial NeuShen
            validation/regression run.

    store : dict, optional
        Keys used to store outputs in state.

    Returns
    -------
    state : dict
        Updated pipeline state containing:
        - cleaned raw object
        - fitted ICA object
        - ICLabel/EOG component summary DataFrame
        - final excluded component indices
        - ICLabel labels and probabilities
        - EOG channels used
        - EOG candidate component indices
        - EOG component scores
        - EOG validation summary
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError(
            "No raw in state. Did you run 'load_eeg' first?"
        )

    # ============================================================
    # Read configuration
    # ============================================================
    ica_kw = params.get("ica", {})
    fit_kw = params.get("fit", {})
    iclabel_kw = params.get("iclabel", {})
    eog_kw = params.get("eog", {})
    store_kw = params.get("store", {})

    if (
        not isinstance(ica_kw, dict)
        or not isinstance(fit_kw, dict)
        or not isinstance(iclabel_kw, dict)
        or not isinstance(eog_kw, dict)
        or not isinstance(store_kw, dict)
    ):
        raise TypeError(
            "step_run_ica_iclabel expects dictionaries for "
            "'ica', 'fit', 'iclabel', 'eog', and 'store'."
        )

    # ============================================================
    # Defaults
    # ============================================================
    ica_defaults: Dict[str, Any] = {
        "n_components": 0.99,
        "method": "infomax",
        "fit_params": {"extended": True},
        "random_state": 42,
        "max_iter": "auto",
    }

    # Generic default remains 100 Hz, but the NeuShen notebook explicitly
    # overrides desired_h_freq to 45 Hz.
    fit_defaults: Dict[str, Any] = {
        "notch_freqs": None,
        "l_freq": 1.0,
        "desired_h_freq": 100.0,
        "apply_average_ref": True,
        "picks": "eeg",
        "skip_by_annotation": (
            "edge",
            "bad_acq_skip",
            "BAD_non_analysis",
        ),
        "reject_by_annotation": True,
    }

    # ICLabel controls automatic artifact-component exclusion.
    iclabel_defaults: Dict[str, Any] = {
        "artifact_labels": [
            "eye blink",
            "muscle artifact",
            "line noise",
            "heart beat",
            "channel noise",
        ],
        "prob_threshold": 0.8,
    }

    # EOG initially provides independent QC evidence only.
    # It does not change exclusions unless use_for_exclusion=True.
    eog_defaults: Dict[str, Any] = {
        "enabled": True,
        "ch_names": None,              # None -> all channels typed as EOG
        "threshold": 3.0,
        "measure": "zscore",
        "l_freq": 1.0,
        "h_freq": 10.0,
        "reject_by_annotation": True,
        "use_for_exclusion": False,
    }

    ica_cfg: Dict[str, Any] = {
        **ica_defaults,
        **ica_kw,
    }

    fit_cfg: Dict[str, Any] = {
        **fit_defaults,
        **fit_kw,
    }

    iclabel_cfg: Dict[str, Any] = {
        **iclabel_defaults,
        **iclabel_kw,
    }

    eog_cfg: Dict[str, Any] = {
        **eog_defaults,
        **eog_kw,
    }

    # ============================================================
    # Output keys
    # ============================================================
    raw_key = str(
        store_kw.get("raw_key", "raw")
    )

    ica_key = str(
        store_kw.get("ica_key", "ica")
    )

    ic_df_key = str(
        store_kw.get("ic_df_key", "iclabel_df")
    )

    exclude_key = str(
        store_kw.get("exclude_key", "excluded_ics")
    )

    labels_key = str(
        store_kw.get("labels_key", "ic_labels")
    )

    proba_key = str(
        store_kw.get("proba_key", "ic_probs")
    )

    # ============================================================
    # 1. Build ICA-fit branch
    # ============================================================
    raw_ic = raw.copy().load_data()

    picks = fit_cfg.get(
        "picks",
        "eeg",
    )

    skip_by_annotation = fit_cfg.get(
        "skip_by_annotation",
        (
            "edge",
            "bad_acq_skip",
            "BAD_non_analysis",
        ),
    )

    # ------------------------------------------------------------
    # Optional notch filtering on ICA-fit copy
    # ------------------------------------------------------------
    notch_freqs = fit_cfg.get(
        "notch_freqs",
        None,
    )

    if notch_freqs is not None:
        if isinstance(
            notch_freqs,
            (int, float),
        ):
            notch_freqs_list = [
                float(notch_freqs)
            ]
        else:
            notch_freqs_list = [
                float(freq)
                for freq in notch_freqs
            ]

        if notch_freqs_list:
            if verbose:
                print(
                    f"→ ICA branch notch filter: "
                    f"{notch_freqs_list}"
                )

            raw_ic.notch_filter(
                freqs=notch_freqs_list,
                picks=picks,
                skip_by_annotation=skip_by_annotation,
                verbose=False,
            )

    # ------------------------------------------------------------
    # ICA-fit bandpass
    # ------------------------------------------------------------
    l_freq = float(
        fit_cfg.get(
            "l_freq",
            1.0,
        )
    )

    desired_h_freq = float(
        fit_cfg.get(
            "desired_h_freq",
            100.0,
        )
    )

    h_freq = _safe_high_cut_for_iclabel(
        sfreq=float(
            raw_ic.info["sfreq"]
        ),
        desired=desired_h_freq,
    )

    if verbose:
        print(
            f"→ ICA branch bandpass: "
            f"{l_freq}–{h_freq} Hz"
        )

    # Only EEG channels are filtered here because ICA is fit on EEG.
    raw_ic.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=picks,
        skip_by_annotation=skip_by_annotation,
        verbose=False,
    )

    # ------------------------------------------------------------
    # Optional average reference on ICA branch
    # ------------------------------------------------------------
    if bool(
        fit_cfg.get(
            "apply_average_ref",
            True,
        )
    ):
        if verbose:
            print(
                "→ ICA branch average reference"
            )

        raw_ic.set_eeg_reference(
            "average",
            verbose=False,
        )

    # ============================================================
    # 2. Fit ICA
    # ============================================================
    if verbose:
        print(
            f"→ Fitting ICA with params: "
            f"{ica_cfg}"
        )

    ica = ICA(
        n_components=ica_cfg["n_components"],
        method=ica_cfg["method"],
        fit_params=ica_cfg["fit_params"],
        random_state=ica_cfg["random_state"],
        max_iter=ica_cfg["max_iter"],
    )

    ica.fit(
        raw_ic,
        picks=picks,
        reject_by_annotation=bool(
            fit_cfg.get(
                "reject_by_annotation",
                True,
            )
        ),
        verbose=False,
    )

    # ============================================================
    # 3. ICLabel classification
    # ============================================================
    ic_labels: Dict[str, Any] = label_components(
        raw_ic,
        ica,
        method="iclabel",
    )

    labels = list(
        ic_labels["labels"]
    )

    y_pred_proba = np.asarray(
        ic_labels["y_pred_proba"],
        dtype=float,
    )

    if len(labels) != len(y_pred_proba):
        raise RuntimeError(
            "ICLabel returned mismatched labels and probabilities: "
            f"len(labels)={len(labels)}, "
            f"len(y_pred_proba)={len(y_pred_proba)}"
        )

    artifact_labels = {
        str(label).lower()
        for label in iclabel_cfg.get(
            "artifact_labels",
            [],
        )
    }

    prob_threshold = iclabel_cfg.get(
        "prob_threshold",
        0.8,
    )

    if prob_threshold is not None:
        prob_threshold = float(
            prob_threshold
        )

    # Components selected by ICLabel alone.
    iclabel_exclude: list[int] = []

    for idx, (label, proba) in enumerate(
        zip(
            labels,
            y_pred_proba,
        )
    ):
        label_lower = str(
            label
        ).lower()

        if label_lower in artifact_labels:
            if (
                prob_threshold is None
                or float(proba) >= prob_threshold
            ):
                iclabel_exclude.append(
                    int(idx)
                )

    # ============================================================
    # 4. EOG-supported ocular-component QC
    # ============================================================
    eog_channels: list[str] = []
    missing_eog_channels: list[str] = []
    eog_candidate_ics: set[int] = set()
    eog_scores_by_channel: dict[str, list[float]] = {}

    if bool(
        eog_cfg.get(
            "enabled",
            True,
        )
    ):
        configured_eog = eog_cfg.get(
            "ch_names",
            None,
        )

        # --------------------------------------------------------
        # Automatically use channels explicitly typed as EOG.
        # --------------------------------------------------------
        if configured_eog is None:
            eog_picks = mne.pick_types(
                raw_ic.info,
                eeg=False,
                eog=True,
                meg=False,
                stim=False,
                misc=False,
                exclude=[],
            )

            eog_channels = [
                raw_ic.ch_names[index]
                for index in eog_picks
            ]

        # --------------------------------------------------------
        # Or use explicitly configured EOG channel names.
        # --------------------------------------------------------
        else:
            if isinstance(
                configured_eog,
                str,
            ):
                requested_eog = [
                    configured_eog
                ]
            else:
                requested_eog = [
                    str(channel)
                    for channel in configured_eog
                ]

            eog_channels = [
                channel
                for channel in requested_eog
                if channel in raw_ic.ch_names
            ]

            missing_eog_channels = [
                channel
                for channel in requested_eog
                if channel not in raw_ic.ch_names
            ]

        # --------------------------------------------------------
        # Score ICA components against each available EOG channel.
        # --------------------------------------------------------
        for eog_channel in eog_channels:
            try:
                eog_indices, eog_scores = (
                    ica.find_bads_eog(
                        raw_ic,
                        ch_name=eog_channel,
                        threshold=float(
                            eog_cfg.get(
                                "threshold",
                                3.0,
                            )
                        ),
                        l_freq=float(
                            eog_cfg.get(
                                "l_freq",
                                1.0,
                            )
                        ),
                        h_freq=float(
                            eog_cfg.get(
                                "h_freq",
                                10.0,
                            )
                        ),
                        reject_by_annotation=bool(
                            eog_cfg.get(
                                "reject_by_annotation",
                                True,
                            )
                        ),
                        measure=str(
                            eog_cfg.get(
                                "measure",
                                "zscore",
                            )
                        ),
                        verbose=False,
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    "EOG-supported ICA component scoring failed "
                    f"for channel '{eog_channel}'."
                ) from exc

            eog_scores = np.asarray(
                eog_scores,
                dtype=float,
            )

            if len(eog_scores) != len(labels):
                raise RuntimeError(
                    "EOG component-score count does not match "
                    "the number of ICA components. "
                    f"Channel={eog_channel}, "
                    f"scores={len(eog_scores)}, "
                    f"components={len(labels)}."
                )

            eog_scores_by_channel[
                eog_channel
            ] = eog_scores.tolist()

            eog_candidate_ics.update(
                int(index)
                for index in eog_indices
            )

    eog_candidate_ics_sorted = sorted(
        eog_candidate_ics
    )

    # ============================================================
    # 5. Determine final component exclusions
    # ============================================================
    # Start with ICLabel selections.
    exclude = sorted(
        set(iclabel_exclude)
    )

    # During initial NeuShen validation this stays False.
    # Therefore EOG is independent QC evidence and does not yet
    # alter the cleaned signal.
    if bool(
        eog_cfg.get(
            "use_for_exclusion",
            False,
        )
    ):
        exclude = sorted(
            set(exclude)
            | set(eog_candidate_ics_sorted)
        )

    # ============================================================
    # 6. Print component-level QC
    # ============================================================
    if verbose:
        print(
            "→ ICLabel + EOG component summary:"
        )

        for idx, (label, proba) in enumerate(
            zip(
                labels,
                y_pred_proba,
            )
        ):
            iclabel_flag = (
                idx in iclabel_exclude
            )

            eog_flag = (
                idx in eog_candidate_ics_sorted
            )

            final_flag = (
                idx in exclude
            )

            status_parts = []

            if iclabel_flag:
                status_parts.append(
                    "ICLabel"
                )

            if eog_flag:
                status_parts.append(
                    "EOG"
                )

            evidence = (
                "+".join(status_parts)
                if status_parts
                else "none"
            )

            final_status = (
                "EXCLUDE"
                if final_flag
                else "keep"
            )

            print(
                f"   IC {idx:03d}: "
                f"{str(label):16s} "
                f"p={float(proba):.3f} | "
                f"evidence={evidence:12s} | "
                f"{final_status}"
            )

        print(
            f"→ EOG channels used: "
            f"{eog_channels or 'None'}"
        )

        if missing_eog_channels:
            print(
                f"→ Requested EOG channels not found: "
                f"{missing_eog_channels}"
            )

        print(
            f"→ ICLabel candidate/excluded ICs: "
            f"{iclabel_exclude}"
        )

        print(
            f"→ EOG candidate ICs: "
            f"{eog_candidate_ics_sorted}"
        )

        print(
            f"→ EOG controls automatic exclusion: "
            f"{bool(eog_cfg.get('use_for_exclusion', False))}"
        )

        print(
            f"→ Final excluded ICs: "
            f"{exclude}"
        )

    # ============================================================
    # 7. Apply ICA back to original raw
    # ============================================================
    ica.exclude = exclude

    raw_clean = raw.copy()

    ica.apply(
        raw_clean,
        exclude=exclude,
        verbose=False,
    )

    # ============================================================
    # 8. Build detailed ICA QC table
    # ============================================================
    ic_df = pd.DataFrame({
        "ic": np.arange(
            len(labels),
            dtype=int,
        ),
        "label": labels,
        "y_pred_proba":
            y_pred_proba.astype(float),
        "iclabel_candidate": [
            idx in iclabel_exclude
            for idx in range(
                len(labels)
            )
        ],
        "eog_candidate": [
            idx in eog_candidate_ics_sorted
            for idx in range(
                len(labels)
            )
        ],
        "excluded": [
            idx in exclude
            for idx in range(
                len(labels)
            )
        ],
    })

    # Add the actual component/EOG score from each recorded EOG channel.
    for eog_channel, scores in (
        eog_scores_by_channel.items()
    ):
        ic_df[
            f"eog_score_{eog_channel}"
        ] = scores

    # ============================================================
    # 9. Store outputs
    # ============================================================
    state[raw_key] = raw_clean
    state[ica_key] = ica
    state[ic_df_key] = ic_df

    # Final components actually removed from the EEG.
    state[exclude_key] = exclude

    # ICLabel outputs.
    state[labels_key] = labels
    state[proba_key] = (
        y_pred_proba
        .astype(float)
        .tolist()
    )

    # Preserve the separate sources of artifact evidence.
    state["iclabel_excluded_ics"] = (
        list(iclabel_exclude)
    )

    state["eog_channels"] = (
        list(eog_channels)
    )

    state["missing_eog_channels"] = (
        list(missing_eog_channels)
    )

    state["eog_candidate_ics"] = (
        list(eog_candidate_ics_sorted)
    )

    state["eog_scores_by_channel"] = (
        dict(eog_scores_by_channel)
    )

    state["eog_validation_summary"] = {
        "enabled": bool(
            eog_cfg.get(
                "enabled",
                True,
            )
        ),
        "eog_available": bool(
            eog_channels
        ),
        "eog_channels":
            list(eog_channels),
        "missing_eog_channels":
            list(missing_eog_channels),
        "eog_candidate_ics":
            list(eog_candidate_ics_sorted),
        "n_eog_candidate_ics":
            int(
                len(
                    eog_candidate_ics_sorted
                )
            ),
        "iclabel_excluded_ics":
            list(iclabel_exclude),
        "final_excluded_ics":
            list(exclude),
        "use_for_exclusion": bool(
            eog_cfg.get(
                "use_for_exclusion",
                False,
            )
        ),
        "threshold": float(
            eog_cfg.get(
                "threshold",
                3.0,
            )
        ),
        "measure": str(
            eog_cfg.get(
                "measure",
                "zscore",
            )
        ),
    }

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
    """Apply band-pass filtering while skipping non-analysis segments."""
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    filter_kw = dict(params)
    filter_kw.setdefault(
        "skip_by_annotation",
        ("edge", "bad_acq_skip", "BAD_non_analysis"),
    )

    if verbose:
        print(f"→ Band-pass filter with params: {filter_kw}")

    raw.filter(**filter_kw)
    state["raw"] = raw
    return state

def step_notch_filter(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """Apply notch filtering while skipping non-analysis segments."""
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    if "freqs" not in params:
        raise ValueError("notch_filter requires 'freqs' in params.")

    filter_kw = dict(params)
    freqs = filter_kw["freqs"]
    freqs_list = [float(freqs)] if isinstance(freqs, (int, float)) else [float(f) for f in freqs]

    sfreq = float(raw.info["sfreq"])
    nyq = sfreq / 2.0
    bad = [f for f in freqs_list if f >= nyq]
    if bad:
        raise ValueError(
            f"Invalid notch freqs {bad}: must be < Nyquist ({nyq:.2f} Hz) "
            f"given current sfreq={sfreq:.2f} Hz."
        )

    filter_kw.setdefault(
        "skip_by_annotation",
        ("edge", "bad_acq_skip", "BAD_non_analysis"),
    )

    if verbose:
        print(f"→ Notch filter with params: {filter_kw}")

    raw.notch_filter(**filter_kw)
    state["raw"] = raw
    return state



# def step_bandpass(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Apply band-pass filtering to the current Raw object in state.

#     Example params:
#       {"l_freq": 0.5, "h_freq": 45.0, "phase": "zero", "fir_design": "firwin"}
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     raw.filter(**params) 


#     state["raw"] = raw
#     return state


# def step_notch_filter(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Apply notch filtering to the current Raw object in state.
#     Requires params["freqs"] to be < Nyquist (sfreq / 2).
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     # --- Hard validation: freqs must be below Nyquist ---
#     if "freqs" not in params:
#         raise ValueError("notch_filter requires 'freqs' in params (e.g., {'freqs': [60, 120]}).")

#     freqs = params["freqs"]
#     if isinstance(freqs, (int, float)):
#         freqs_list = [float(freqs)]
#     else:
#         freqs_list = [float(f) for f in freqs]

#     sfreq = float(raw.info["sfreq"])
#     nyq = sfreq / 2.0
#     bad = [f for f in freqs_list if f >= nyq]
#     if bad:
#         msg = (
#             f"Invalid notch freqs {bad}: must be < Nyquist ({nyq:.2f} Hz) "
#             f"given current sfreq={sfreq:.2f} Hz. "
#             f"Resample higher or choose lower freqs."
#         )
#         print(msg)
#         raise ValueError(msg)

#     if verbose:
#         print(f"→ Notch filter with params: {params}")

#     raw.notch_filter(**params)  # let MNE validate remaining details
#     state["raw"] = raw
#     return state




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


# def step_plot_raw(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Plot the raw time-series browser using MNE: raw.plot(**params)

#     Example params:
#       {"n_channels": 32, "picks": "eeg", "duration": 10.0}
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     if verbose:
#         print(f"→ raw.plot with params: {params}")

#     raw.plot(**params)
#     return state


def step_plot_raw(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Plot the raw time-series browser using MNE without blocking the pipeline.

    Example params:
      {"n_channels": 32, "picks": "eeg", "duration": 10.0, "block": False}
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    plot_kw = dict(params)
    plot_kw.setdefault("block", False)  # Continue pipeline while browser remains open

    if verbose:
        print(f"→ raw.plot with params: {plot_kw}")

    raw.plot(**plot_kw)
    return state



# def step_plot_psd(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Compute and plot PSD in one shot:
#         raw.compute_psd(**params).plot()
#         plt.show()

#     params are passed ONLY to raw.compute_psd.
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     if verbose:
#         print(f"→ raw.compute_psd with params: {params}")

#     raw.compute_psd(**params).plot()
#     plt.show()

#     return state


def step_plot_psd(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute and plot the raw EEG PSD without blocking the preprocessing pipeline.

    Important
    ---------
    'average' controls CHANNEL averaging in Spectrum.plot(), not Welch-segment
    averaging in raw.compute_psd().

    'block' controls whether matplotlib stops pipeline execution.

    All remaining parameters are passed to raw.compute_psd().
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    psd_kw = dict(params)

    # Plot-only parameters; do NOT pass these into raw.compute_psd().
    plot_average = bool(psd_kw.pop("average", False))
    block = bool(psd_kw.pop("block", False))

    # Welch should average its time segments normally.
    psd_kw.setdefault("method", "welch")

    if verbose:
        print(f"→ raw.compute_psd with params: {psd_kw}")
        print(f"→ Spectrum plot: average={plot_average}, block={block}")

    spectrum = raw.compute_psd(**psd_kw)

    # average=False here means show individual CHANNEL spectra.
    spectrum.plot(
        average=plot_average,
        show=False,
    )

    plt.show(block=block)
    return state



def step_mad_bad_channels(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Detect and interpolate extreme high-variability EEG channels using
    a MAD-based modified z-score.

    This is a supplemental bad-channel detection step that runs before
    RANSAC and before average referencing.

    Workflow
    --------
    1. Calculate temporal standard deviation (SD) for each EEG channel.
    2. Calculate the median SD across all EEG channels.
    3. Calculate the median absolute deviation (MAD) of those channel SDs.
    4. Calculate the modified z-score for each channel:

           modified_z =
               0.67448975 * (channel_sd - median_sd) / MAD

    5. Flag unusually HIGH-variability channels whose modified z-score
       exceeds the configured threshold.
    6. Interpolate the detected channels using MNE.

    Parameters
    ----------
    threshold : float
        Modified-z threshold for identifying an outlier.
        Default: 3.5.

    reject_by_annotation : str or None
        Controls whether BAD-annotated samples are omitted when calculating
        channel SD. Default: "omit".

    reset_bads : bool
        Clear MNE bad-channel flags after interpolation.
        Default: True.

    Returns
    -------
    state : dict
        Adds:
        - state["mad_bad_channels"]
        - state["mad_channel_qc_df"]
        - state["mad_channel_summary"]
    """

    raw = state.get("raw")

    if raw is None:
        raise RuntimeError(
            "No raw in state. Did you run 'load_eeg' first?"
        )

    # ============================================================
    # Read configuration
    # ============================================================
    threshold = float(
        params.get(
            "threshold",
            3.5,
        )
    )

    reject_by_annotation = params.get(
        "reject_by_annotation",
        "omit",
    )

    reset_bads = bool(
        params.get(
            "reset_bads",
            True,
        )
    )

    if threshold <= 0:
        raise ValueError(
            "MAD threshold must be > 0."
        )

    # ============================================================
    # Select EEG channels only
    # ============================================================
    eeg_picks = mne.pick_types(
        raw.info,
        eeg=True,
        meg=False,
        eog=False,
        stim=False,
        misc=False,
        exclude=[],
    )

    if len(eeg_picks) == 0:
        raise RuntimeError(
            "No EEG channels available for MAD bad-channel detection."
        )

    channel_names = [
        raw.ch_names[index]
        for index in eeg_picks
    ]

    # ============================================================
    # Get EEG data before interpolation / average reference.
    #
    # BAD_non_analysis samples are omitted when configured.
    # ============================================================
    eeg_data = raw.get_data(
        picks=eeg_picks,
        reject_by_annotation=reject_by_annotation,
    )

    if eeg_data.shape[1] == 0:
        raise RuntimeError(
            "No EEG samples available for MAD bad-channel detection."
        )

    # ============================================================
    # 1. Calculate temporal SD for every EEG channel
    # ============================================================
    channel_sd_uv = (
        np.std(
            eeg_data,
            axis=1,
        )
        * 1e6
    )

    # ============================================================
    # 2. Calculate median channel SD
    # ============================================================
    median_sd_uv = float(
        np.median(
            channel_sd_uv
        )
    )

    # ============================================================
    # 3. Calculate median absolute deviation (MAD)
    # ============================================================
    mad_sd_uv = float(
        np.median(
            np.abs(
                channel_sd_uv
                - median_sd_uv
            )
        )
    )

    # ============================================================
    # 4. Calculate MAD-based modified z-score
    # ============================================================
    if mad_sd_uv > 0:

        modified_z = (
            0.67448975
            * (
                channel_sd_uv
                - median_sd_uv
            )
            / mad_sd_uv
        )

    else:

        # No meaningful outlier separation if MAD is zero.
        modified_z = np.zeros_like(
            channel_sd_uv,
            dtype=float,
        )

    # ============================================================
    # 5. Detect HIGH-variability outliers only
    # ============================================================
    outlier_mask = (
        modified_z
        > threshold
    )

    mad_bad_channels = [
        channel_names[index]
        for index in np.where(
            outlier_mask
        )[0]
    ]

    # ============================================================
    # Store detailed QC information BEFORE interpolation
    # ============================================================
    mad_channel_qc_df = pd.DataFrame({
        "channel": channel_names,
        "temporal_sd_uv":
            channel_sd_uv.astype(float),
        "median_channel_sd_uv":
            median_sd_uv,
        "mad_channel_sd_uv":
            mad_sd_uv,
        "modified_z":
            modified_z.astype(float),
        "threshold":
            threshold,
        "mad_outlier":
            outlier_mask.astype(bool),
    })

    mad_channel_qc_df = (
        mad_channel_qc_df
        .sort_values(
            "modified_z",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ============================================================
    # 6. Interpolate MAD-detected bad channels
    # ============================================================
    raw.info["bads"] = list(
        mad_bad_channels
    )

    if mad_bad_channels:

        raw.interpolate_bads(
            reset_bads=reset_bads,
        )

    # ============================================================
    # Store outputs for QC / traceability
    # ============================================================
    state["mad_bad_channels"] = list(
        mad_bad_channels
    )

    state["mad_channel_qc_df"] = (
        mad_channel_qc_df
    )

    state["mad_channel_summary"] = {
        "method": "mad_modified_z",
        "metric": "temporal_standard_deviation",
        "threshold": threshold,
        "median_channel_sd_uv": median_sd_uv,
        "mad_channel_sd_uv": mad_sd_uv,
        "n_eeg_channels": int(
            len(channel_names)
        ),
        "n_bad_channels": int(
            len(mad_bad_channels)
        ),
        "bad_channels": list(
            mad_bad_channels
        ),
    }

    state["raw"] = raw

    if verbose:
        print("\nMAD bad-channel detection + interpolation")
        print("-" * 50)
        print(
            f"Median channel SD: "
            f"{median_sd_uv:.3f} µV"
        )
        print(
            f"MAD of channel SD: "
            f"{mad_sd_uv:.3f} µV"
        )
        print(
            f"Modified-z threshold: "
            f"{threshold}"
        )
        print(
            f"MAD bad channels: "
            f"{mad_bad_channels}"
        )

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

        # Ensure RANSAC ignores epochs overlapping BAD annotations
        epochs_kw = dict(epochs_kw)
        epochs_kw.setdefault("reject_by_annotation", True)

        if "tmax" not in epochs_kw:
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




# def step_fixed_length_epochs(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False
# ) -> Dict[str, Any]:
#     """
#     Create fixed-length events and epochs from the current Raw.

#     Params (recommended structure)
#     ------------------------------
#     {
#       "events": {...},   # kwargs for mne.make_fixed_length_events(raw, **events)
#       "epochs": {...},   # kwargs for mne.Epochs(raw, events, **epochs)
#       "store": {         # optional
#           "events_key": "events",
#           "epochs_key": "epochs"
#       }
#     }

#     Minimal example
#     ---------------
#     {
#       "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
#       "epochs": {"event_id": {"seg": 2}, "tmin": 0.0, "tmax": 2.0, "baseline": None,
#                 "reject": None, "detrend": 0, "preload": True}
#     }
#     """
#     raw = state.get("raw")
#     if raw is None:
#         raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

#     events_kw = params.get("events", {})
#     epochs_kw = params.get("epochs", {})
#     store_kw = params.get("store", {})

#     if not isinstance(events_kw, dict) or not isinstance(epochs_kw, dict):
#         raise TypeError("fixed_length_epochs expects dicts for params['events'] and params['epochs'].")

#     # ---- events ----
#     if "duration" not in events_kw:
#         raise ValueError("fixed_length_epochs requires params['events']['duration'].")

#     if verbose:
#         print(f"→ Fixed-length events params: {events_kw}")

#     events = mne.make_fixed_length_events(raw, **events_kw)

#     # ---- epochs ----
#     # Sensible default: if user omitted tmax, use duration
#     if "tmax" not in epochs_kw:
#         epochs_kw = dict(epochs_kw)
#         epochs_kw["tmax"] = float(events_kw["duration"])

#     if verbose:
#         print(f"→ Epochs params: {epochs_kw}")

#     epochs = mne.Epochs(raw, events, **epochs_kw)

#     # ---- store ----
#     events_key = store_kw.get("events_key", "events")
#     epochs_key = store_kw.get("epochs_key", "epochs")

#     state[events_key] = events
#     state[epochs_key] = epochs

#     if verbose:
#         print(f"→ Stored events in state['{events_key}'], epochs in state['{epochs_key}']")

#     return state


def step_fixed_length_epochs(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Create fixed-length epochs.

    If recognized analysis conditions exist, epochs are created separately
    within each condition block. Otherwise, the original whole-recording
    fixed-length behavior is used.
    """
    raw = state.get("raw")
    if raw is None:
        raise RuntimeError("No raw in state. Did you run 'load_eeg' first?")

    events_kw = dict(params.get("events", {}))
    epochs_kw = dict(params.get("epochs", {}))
    store_kw = dict(params.get("store", {}))
    if "duration" not in events_kw:
        raise ValueError("fixed_length_epochs requires params['events']['duration'].")

    duration = float(events_kw["duration"])
    overlap = float(events_kw.get("overlap", 0.0))
    event_code = int(events_kw.get("id", 2))
    if duration <= 0:
        raise ValueError("Epoch duration must be > 0.")
    if overlap < 0 or overlap >= duration:
        raise ValueError("Epoch overlap must satisfy 0 <= overlap < duration.")

    sfreq = float(raw.info["sfreq"])
    duration_samples = int(round(duration * sfreq))
    step_samples = int(round((duration - overlap) * sfreq))

    # Exact N-sample epoch by default; e.g. 2 s at 250 Hz = 500 samples.
    epochs_kw.setdefault("tmin", 0.0)
    epochs_kw.setdefault("tmax", (duration_samples - 1) / sfreq)
    epochs_kw.setdefault("reject_by_annotation", True)

    conditions_key = str(params.get("conditions_key", "analysis_conditions"))
    use_conditions = params.get("use_analysis_conditions", "auto")
    conditions = state.get(conditions_key, {})

    if use_conditions == "auto":
        condition_mode = isinstance(conditions, Mapping) and bool(conditions)
    elif isinstance(use_conditions, bool):
        condition_mode = use_conditions
    else:
        raise ValueError("use_analysis_conditions must be True, False, or 'auto'.")

    if condition_mode and not conditions:
        raise RuntimeError(
            f"Condition-aware epoching requested but state['{conditions_key}'] is empty."
        )

    # ------------------------------------------------------------------
    # CONDITION-AWARE MODE
    # ------------------------------------------------------------------
    if condition_mode:
        if float(epochs_kw.get("tmin", 0.0)) != 0.0:
            raise ValueError("Condition-aware fixed-length epoching currently requires tmin=0.")

        events_by_condition = {}
        epochs_by_condition = {}
        summary = {}

        for condition, blocks in conditions.items():
            starts = []

            # Restart segmentation at the beginning of every condition block.
            for block in blocks:
                start = max(0, int(block["start_sample"]))
                stop = min(int(raw.n_times), int(block["stop_sample"]))
                last_start = stop - duration_samples
                if last_start >= start:
                    starts.extend(range(start, last_start + 1, step_samples))

            if not starts:
                summary[str(condition)] = {
                    "n_blocks": len(blocks),
                    "n_epochs_attempted": 0,
                    "n_epochs_created": 0,
                }
                continue

            # MNE event samples include raw.first_samp.
            event_samples = np.asarray(starts, dtype=int) + int(raw.first_samp)
            events = np.column_stack([
                event_samples,
                np.zeros(len(event_samples), dtype=int),
                np.full(len(event_samples), event_code, dtype=int),
            ])

            ep_kw = dict(epochs_kw)
            ep_kw.setdefault("event_id", {str(condition): event_code})
            epochs = mne.Epochs(raw, events, **ep_kw)

            events_by_condition[str(condition)] = events
            epochs_by_condition[str(condition)] = epochs
            summary[str(condition)] = {
                "n_blocks": len(blocks),
                "n_epochs_attempted": len(events),
                "n_epochs_created": len(epochs),
                "duration_sec": duration,
            }

        events_key = str(store_kw.get("events_key", "events_final")) + "_by_condition"
        epochs_key = str(store_kw.get("epochs_key", "epochs_final")) + "_by_condition"

        state[events_key] = events_by_condition
        state[epochs_key] = epochs_by_condition
        state["condition_epoch_summary"] = summary
        state["epoch_mode"] = "condition"

        if verbose:
            print("\nCondition-specific fixed-length epoching")
            print("-" * 40)
            for condition, info in summary.items():
                print(
                    f"{condition}: {info['n_blocks']} block(s), "
                    f"{info['n_epochs_attempted']} attempted, "
                    f"{info['n_epochs_created']} created"
                )
            print(f"Stored epochs in state['{epochs_key}']")

        return state

    # ------------------------------------------------------------------
    # NORMAL FALLBACK MODE
    # ------------------------------------------------------------------
    if verbose:
        print("→ No analysis conditions used; applying normal fixed-length epoching.")

    events = mne.make_fixed_length_events(raw, **events_kw)
    epochs = mne.Epochs(raw, events, **epochs_kw)

    events_key = store_kw.get("events_key", "events")
    epochs_key = store_kw.get("epochs_key", "epochs")
    state[events_key] = events
    state[epochs_key] = epochs
    state["epoch_mode"] = "single"

    if verbose:
        print(f"→ Stored events in state['{events_key}'], epochs in state['{epochs_key}']")

    return state





# def step_reject_bad_epochs(
#     state: Dict[str, Any],
#     params: Dict[str, Any],
#     verbose: bool = False,
# ) -> Dict[str, Any]:
#     """
#     Detect and remove residual bad epochs after the main EEG cleaning steps.

#     This step:
#       1. Reads an existing Epochs object from the pipeline state.
#       2. Estimates a global EEG peak-to-peak rejection threshold using
#          autoreject.get_rejection_threshold(), unless a manual threshold
#          dictionary is supplied.
#       3. Drops epochs exceeding the threshold.
#       4. Preserves the original epochs and stores the cleaned epochs separately.
#       5. Stores a summary, rejection threshold, dropped epoch indices,
#          and drop reasons in the pipeline state.

#     Expected params
#     ---------------
#     epochs_key : str
#         State key containing the input epochs.
#         Default: "epochs_final"

#     output_key : str
#         State key used for the cleaned epochs.
#         Default: "epochs_clean"

#     reject : dict or None
#         Manual peak-to-peak rejection threshold dictionary.

#         Example:
#             {"eeg": 150e-6}

#         EEG values must be expressed in volts.

#         If None, the threshold is estimated automatically from the recording.

#     flat : dict or None
#         Optional minimum peak-to-peak threshold for detecting nearly flat
#         epochs. Leave as None initially.

#     ch_types : str or list[str]
#         Channel type used when estimating the rejection threshold.
#         Default: "eeg"

#     cv : int
#         Number of cross-validation folds used to estimate the threshold.
#         Default: 5

#     random_state : int
#         Random seed for reproducibility.
#         Default: 42

#     decim : int
#         Decimation used only while estimating the threshold.
#         Default: 1

#     Returns
#     -------
#     state : dict
#         Adds:
#           - state[output_key]
#           - state["bad_epoch_reject_thresholds"]
#           - state["bad_epoch_indices"]
#           - state["bad_epoch_drop_reasons"]
#           - state["bad_epoch_rejection_summary"]
#     """
#     epochs_key = str(params.get("epochs_key", "epochs_final"))
#     output_key = str(params.get("output_key", "epochs_clean"))

#     epochs_source = state.get(epochs_key)

#     if epochs_source is None:
#         raise RuntimeError(
#             f"No epochs found in state['{epochs_key}']. "
#             "Run fixed_length_epochs before reject_bad_epochs."
#         )

#     if not isinstance(epochs_source, mne.BaseEpochs):
#         raise TypeError(
#             f"state['{epochs_key}'] must contain an MNE Epochs object. "
#             f"Got {type(epochs_source).__name__}."
#         )

#     # Work on a copy so epochs_final remains unchanged for comparison.
#     epochs_clean = epochs_source.copy().load_data()

#     n_before = len(epochs_clean)
#     if n_before < 2:
#         raise RuntimeError(
#             f"At least 2 epochs are needed for bad-epoch rejection. "
#             f"Found {n_before}."
#         )

#     selection_before = epochs_clean.selection.copy()

#     manual_reject = params.get("reject", None)
#     flat = params.get("flat", None)

#     # ------------------------------------------------------------
#     # Determine the peak-to-peak rejection threshold
#     # ------------------------------------------------------------
#     if manual_reject is None:
#         requested_cv = int(params.get("cv", 5))
#         cv = min(requested_cv, n_before)

#         if cv < 2:
#             raise RuntimeError(
#                 f"Cross-validation requires at least 2 folds. Got cv={cv}."
#             )

#         reject = get_rejection_threshold(
#             epochs_clean,
#             decim=int(params.get("decim", 1)),
#             random_state=params.get("random_state", 42),
#             ch_types=params.get("ch_types", "eeg"),
#             cv=cv,
#             verbose=verbose,
#         )

#         threshold_source = "automatically estimated"
#     else:
#         if not isinstance(manual_reject, dict):
#             raise TypeError(
#                 "params['reject'] must be a dictionary such as "
#                 "{'eeg': 150e-6}, or None."
#             )

#         reject = {
#             str(channel_type): float(value)
#             for channel_type, value in manual_reject.items()
#         }

#         threshold_source = "manually supplied"

#     if not reject:
#         raise RuntimeError(
#             "No rejection threshold was produced. Confirm that the epochs "
#             "contain channels of the requested type."
#         )

#     if flat is not None:
#         if not isinstance(flat, dict):
#             raise TypeError(
#                 "params['flat'] must be a dictionary or None."
#             )

#         flat = {
#             str(channel_type): float(value)
#             for channel_type, value in flat.items()
#         }

#     # ------------------------------------------------------------
#     # Drop bad epochs
#     # ------------------------------------------------------------
#     epochs_clean.drop_bad(
#         reject=reject,
#         flat=flat,
#         verbose=verbose,
#     )

#     selection_after = epochs_clean.selection.copy()

#     dropped_epoch_indices = sorted(
#         set(selection_before.tolist()) - set(selection_after.tolist())
#     )

#     drop_reasons = {
#         int(epoch_index): list(epochs_clean.drop_log[epoch_index])
#         for epoch_index in dropped_epoch_indices
#     }

#     n_after = len(epochs_clean)
#     n_rejected = n_before - n_after
#     percent_rejected = (
#         100.0 * n_rejected / n_before if n_before > 0 else 0.0
#     )
#     percent_retained = (
#         100.0 * n_after / n_before if n_before > 0 else 0.0
#     )

#     summary = {
#         "input_epochs_key": epochs_key,
#         "output_epochs_key": output_key,
#         "n_epochs_before": int(n_before),
#         "n_epochs_rejected": int(n_rejected),
#         "n_epochs_retained": int(n_after),
#         "percent_rejected": float(percent_rejected),
#         "percent_retained": float(percent_retained),
#         "threshold_source": threshold_source,
#         "reject_thresholds": {
#             key: float(value)
#             for key, value in reject.items()
#         },
#         "flat_thresholds": flat,
#     }

#     # Store outputs without overwriting epochs_final.
#     state[output_key] = epochs_clean
#     state["bad_epoch_reject_thresholds"] = reject
#     state["bad_epoch_indices"] = dropped_epoch_indices
#     state["bad_epoch_drop_reasons"] = drop_reasons
#     state["bad_epoch_rejection_summary"] = summary

#     if verbose:
#         print("\nBad-epoch rejection summary")
#         print("-" * 40)
#         print(f"Threshold source: {threshold_source}")

#         for channel_type, threshold in reject.items():
#             if channel_type in {"eeg", "eog", "ecg"}:
#                 print(
#                     f"{channel_type.upper()} threshold: "
#                     f"{threshold * 1e6:.2f} microvolts"
#                 )
#             else:
#                 print(
#                     f"{channel_type} threshold: {threshold}"
#                 )

#         print(f"Epochs before:   {n_before}")
#         print(f"Epochs rejected: {n_rejected}")
#         print(f"Epochs retained: {n_after}")
#         print(f"Percent rejected: {percent_rejected:.2f}%")
#         print(f"Percent retained: {percent_retained:.2f}%")

#     return state


def step_reject_bad_epochs(
    state: Dict[str, Any],
    params: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Detect and remove residual bad epochs.

    Automatically supports:
      1. Single Epochs object.
      2. Condition-specific mapping of Epochs objects.

    Automatic rejection thresholds are estimated independently for each
    condition so condition-specific EEG amplitude behavior is preserved.
    """
    epochs_key = str(params.get("epochs_key", "epochs_final"))
    output_key = str(params.get("output_key", "epochs_clean"))
    condition_epochs_key = str(params.get("condition_epochs_key", f"{epochs_key}_by_condition"))
    epochs_source = state.get(epochs_key)
    condition_source = state.get(condition_epochs_key)

    if epochs_source is None and condition_source is None:
        raise RuntimeError(
            f"No epochs found in state['{epochs_key}'] or "
            f"state['{condition_epochs_key}']. Run fixed_length_epochs first."
        )

    manual_reject = params.get("reject", None)
    flat = params.get("flat", None)
    if manual_reject is not None and not isinstance(manual_reject, dict):
        raise TypeError("params['reject'] must be a dictionary or None.")
    if flat is not None and not isinstance(flat, dict):
        raise TypeError("params['flat'] must be a dictionary or None.")

    if manual_reject is not None:
        manual_reject = {str(k): float(v) for k, v in manual_reject.items()}
    if flat is not None:
        flat = {str(k): float(v) for k, v in flat.items()}

    # Clean one Epochs object using the existing rejection logic.
    def _clean_one(epochs: mne.BaseEpochs, label: str | None = None):
        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError(
                f"Expected an MNE Epochs object"
                f"{f' for {label}' if label else ''}; got {type(epochs).__name__}."
            )

        epochs_clean = epochs.copy().load_data()
        n_before = len(epochs_clean)
        if n_before < 2:
            raise RuntimeError(
                f"At least 2 epochs are required"
                f"{f' for {label}' if label else ''}. Found {n_before}."
            )

        selection_before = epochs_clean.selection.copy()

        # Estimate threshold independently unless a manual threshold was supplied.
        if manual_reject is None:
            requested_cv = int(params.get("cv", 5))
            cv = min(requested_cv, n_before)
            if cv < 2:
                raise RuntimeError(f"Cross-validation requires at least 2 folds. Got cv={cv}.")

            reject = get_rejection_threshold(
                epochs_clean,
                decim=int(params.get("decim", 1)),
                random_state=params.get("random_state", 42),
                ch_types=params.get("ch_types", "eeg"),
                cv=cv,
                verbose=verbose,
            )
            threshold_source = "automatically estimated"
        else:
            reject = dict(manual_reject)
            threshold_source = "manually supplied"

        if not reject:
            raise RuntimeError(
                f"No rejection threshold was produced"
                f"{f' for {label}' if label else ''}."
            )

        epochs_clean.drop_bad(reject=reject, flat=flat, verbose=verbose)
        selection_after = epochs_clean.selection.copy()

        dropped = sorted(
            set(selection_before.tolist()) - set(selection_after.tolist())
        )
        reasons = {
            int(idx): list(epochs_clean.drop_log[idx])
            for idx in dropped
        }

        n_after = len(epochs_clean)
        n_rejected = n_before - n_after
        summary = {
            "n_epochs_before": int(n_before),
            "n_epochs_rejected": int(n_rejected),
            "n_epochs_retained": int(n_after),
            "percent_rejected": float(100.0 * n_rejected / n_before),
            "percent_retained": float(100.0 * n_after / n_before),
            "threshold_source": threshold_source,
            "reject_thresholds": {k: float(v) for k, v in reject.items()},
            "flat_thresholds": flat,
        }
        return epochs_clean, reject, dropped, reasons, summary

    # ------------------------------------------------------------------
    # CONDITION-AWARE MODE
    # ------------------------------------------------------------------
    if isinstance(condition_source, Mapping) and condition_source:
        clean_by_condition = {}
        thresholds_by_condition = {}
        indices_by_condition = {}
        reasons_by_condition = {}
        summaries_by_condition = {}

        if verbose:
            print("\nCondition-specific bad-epoch rejection")
            print("-" * 40)

        for condition, epochs in condition_source.items():
            clean, reject, dropped, reasons, summary = _clean_one(
                epochs, label=str(condition)
            )
            clean_by_condition[str(condition)] = clean
            thresholds_by_condition[str(condition)] = reject
            indices_by_condition[str(condition)] = dropped
            reasons_by_condition[str(condition)] = reasons
            summaries_by_condition[str(condition)] = summary

            if verbose:
                threshold_text = ", ".join(
                    f"{k.upper()}={v * 1e6:.2f} µV" if k in {"eeg", "eog", "ecg"}
                    else f"{k}={v}"
                    for k, v in reject.items()
                )
                print(
                    f"{condition}: {summary['n_epochs_before']} before → "
                    f"{summary['n_epochs_retained']} retained "
                    f"({summary['percent_rejected']:.2f}% rejected) | "
                    f"{threshold_text}"
                )

        state[f"{output_key}_by_condition"] = clean_by_condition
        state["bad_epoch_reject_thresholds_by_condition"] = thresholds_by_condition
        state["bad_epoch_indices_by_condition"] = indices_by_condition
        state["bad_epoch_drop_reasons_by_condition"] = reasons_by_condition
        state["bad_epoch_rejection_summary_by_condition"] = summaries_by_condition
        return state

    # ------------------------------------------------------------------
    # ORIGINAL SINGLE-EPOCH MODE
    # ------------------------------------------------------------------
    clean, reject, dropped, reasons, summary = _clean_one(epochs_source)

    summary["input_epochs_key"] = epochs_key
    summary["output_epochs_key"] = output_key
    state[output_key] = clean
    state["bad_epoch_reject_thresholds"] = reject
    state["bad_epoch_indices"] = dropped
    state["bad_epoch_drop_reasons"] = reasons
    state["bad_epoch_rejection_summary"] = summary

    if verbose:
        print("\nBad-epoch rejection summary")
        print("-" * 40)
        print(f"Epochs before:   {summary['n_epochs_before']}")
        print(f"Epochs rejected: {summary['n_epochs_rejected']}")
        print(f"Epochs retained: {summary['n_epochs_retained']}")
        print(f"Percent rejected: {summary['percent_rejected']:.2f}%")

    return state



def step_validate_alternating_condition_annotations(
    state,
    params,
    verbose=False,
):
    """
    Validate annotations for a two-condition alternating analysis design.

    This step is intended for recordings in which two analysis conditions
    are expected to alternate over time, such as:

        EC -> EO -> EC -> EO -> ...

    The function does NOT assume a study-specific interval duration.

    Instead, it:

    1. Reads the complete MNE annotation timeline.
    2. Identifies the two requested analysis-condition annotations.
    3. Checks whether consecutive condition markers alternate.
    4. Uses only structurally valid alternating transitions to learn the
       typical interval duration within that recording.
    5. Uses a robust median/MAD approach to identify unusual interval
       durations.
    6. Preserves only intervals that can be interpreted confidently.
    7. Records ambiguous intervals for QC rather than guessing their
       intended condition.

    Importantly, annotation problems are treated as recording-level data
    issues rather than Python exceptions. If the annotation structure cannot
    be resolved confidently, the function stores:

        state["annotation_validation_status"] = "cannot_calculate"

    together with an explanatory error message and QC information.

    Parameters
    ----------
    state : dict
        Pipeline state containing an MNE Raw object under state["raw"].

    params : dict
        Configuration dictionary.

        Required
        --------
        condition_map : mapping
            Mapping from annotation descriptions to standardized analysis
            conditions.

            Example:

                {
                    "Eyes Open": "EO",
                    "Eyes Closed": "EC",
                }

            Exactly two unique output conditions are required because this
            validator is specifically for alternating two-condition designs.

        Optional
        --------
        timing_modified_z_threshold : float
            Robust modified-z threshold used to identify interval-duration
            outliers after the reference duration has been learned from
            valid alternating transitions.

            Default = 3.5.

            This is NOT an expected duration in seconds. It controls only
            the sensitivity of the data-driven outlier detector.

        min_reference_intervals : int
            Minimum number of structurally valid alternating intervals
            required to learn a recording-specific timing reference.

            Default = 3.

    verbose : bool
        Whether to print a concise annotation-validation summary.

    Returns
    -------
    state : dict
        Updated pipeline state containing:

        annotation_validation_status
            "ok" or "cannot_calculate"

        annotation_validation_error
            None when successful; explanatory text otherwise.

        annotation_timeline_df
            Complete annotation timeline for traceability.

        annotation_validation_qc_df
            One row per candidate analysis-condition interval showing
            structural and timing validation results.

        annotation_validation_summary
            Recording-level validation summary.

        validated_condition_intervals
            Mapping from standardized condition name to accepted intervals.

            Example:

                {
                    "EO": [
                        {"onset": 382.4, "duration": 60.6},
                        ...
                    ],
                    "EC": [
                        {"onset": 312.8, "duration": 69.6},
                        ...
                    ],
                }

    Notes
    -----
    - The function never changes EC to EO or EO to EC.
    - Repeated condition markers are treated as structurally ambiguous.
    - Non-condition annotations such as Movement or Talking are retained
      in QC but do not automatically determine an eye state.
    - Ambiguous intervals are excluded rather than repaired by assumption.
    - No fixed study-specific interval duration or total recording duration
      is required.
    """

    # ------------------------------------------------------------------
    # 0) Validate inputs
    # ------------------------------------------------------------------
    if not isinstance(state, dict):
        raise TypeError(
            "state must be a dictionary."
        )

    raw = state.get("raw")

    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(
            "state['raw'] must contain an MNE Raw object."
        )

    if params is None:
        params = {}

    if not isinstance(params, Mapping):
        raise TypeError(
            "params must be a mapping."
        )

    condition_map = params.get(
        "condition_map"
    )

    if not isinstance(condition_map, Mapping):
        raise ValueError(
            "params['condition_map'] must be provided as a mapping."
        )

    if not condition_map:
        raise ValueError(
            "params['condition_map'] cannot be empty."
        )

    timing_modified_z_threshold = float(
        params.get(
            "timing_modified_z_threshold",
            3.5,
        )
    )

    if timing_modified_z_threshold <= 0:
        raise ValueError(
            "timing_modified_z_threshold must be greater than zero."
        )

    min_reference_intervals = int(
        params.get(
            "min_reference_intervals",
            3,
        )
    )

    if min_reference_intervals < 1:
        raise ValueError(
            "min_reference_intervals must be at least 1."
        )

    # ------------------------------------------------------------------
    # 1) Normalize the requested condition mapping
    # ------------------------------------------------------------------
    #
    # Annotation matching is case-insensitive and ignores leading/trailing
    # whitespace, while the standardized output labels remain exactly as
    # supplied in condition_map.
    # ------------------------------------------------------------------
    normalized_condition_map = {
        str(annotation_name).strip().casefold():
            str(condition_name).strip()

        for annotation_name, condition_name
        in condition_map.items()
    }

    condition_names = list(
        dict.fromkeys(
            normalized_condition_map.values()
        )
    )

    # This validator is intentionally designed for two-condition
    # alternating paradigms. The condition names themselves are generic.
    if len(condition_names) != 2:
        raise ValueError(
            "Alternating-condition annotation validation requires exactly "
            "two unique standardized conditions. "
            f"Received: {condition_names}"
        )

    # ------------------------------------------------------------------
    # 2) Initialize failure-safe outputs
    # ------------------------------------------------------------------
    #
    # These are written before validation begins so downstream QC can
    # always inspect the annotation-validation result.
    # ------------------------------------------------------------------
    state["annotation_validation_status"] = (
        "cannot_calculate"
    )

    state["annotation_validation_error"] = None

    state["annotation_timeline_df"] = (
        pd.DataFrame()
    )

    state["annotation_validation_qc_df"] = (
        pd.DataFrame()
    )

    state["validated_condition_intervals"] = {
        condition: []
        for condition in condition_names
    }

    state["annotation_validation_summary"] = {
        "status": "cannot_calculate",
        "error": None,
        "conditions": list(condition_names),
        "timing_method": "modified_z_from_median_mad",
        "timing_modified_z_threshold":
            timing_modified_z_threshold,
        "min_reference_intervals":
            min_reference_intervals,
    }

    # Small internal helper used only to record a data-level
    # "cannot calculate" result without raising an exception.
    def _cannot_calculate(
        message,
        *,
        timeline_df=None,
        qc_df=None,
        summary_updates=None,
    ):
        state["annotation_validation_status"] = (
            "cannot_calculate"
        )

        state["annotation_validation_error"] = (
            str(message)
        )

        if isinstance(timeline_df, pd.DataFrame):
            state["annotation_timeline_df"] = (
                timeline_df.copy()
            )

        if isinstance(qc_df, pd.DataFrame):
            state["annotation_validation_qc_df"] = (
                qc_df.copy()
            )

        summary = dict(
            state["annotation_validation_summary"]
        )

        summary.update({
            "status": "cannot_calculate",
            "error": str(message),
        })

        if summary_updates:
            summary.update(
                dict(summary_updates)
            )

        state["annotation_validation_summary"] = (
            summary
        )

        state["validated_condition_intervals"] = {
            condition: []
            for condition in condition_names
        }

        if verbose:
            print(
                "[annotation validation] "
                "Status: CANNOT CALCULATE"
            )
            print(
                f"[annotation validation] Reason: {message}"
            )

        return state

    # ------------------------------------------------------------------
    # 3) Read the complete MNE annotation timeline
    # ------------------------------------------------------------------
    annotations = raw.annotations

    if annotations is None or len(annotations) == 0:
        return _cannot_calculate(
            "Recording contains no annotations."
        )

    timeline_rows = []

    for annotation_idx, (
        onset,
        duration,
        description,
    ) in enumerate(
        zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        )
    ):

        description = str(
            description
        )

        normalized_description = (
            description
            .strip()
            .casefold()
        )

        condition = (
            normalized_condition_map.get(
                normalized_description
            )
        )

        timeline_rows.append({
            "annotation_idx":
                int(annotation_idx),

            "onset_sec":
                float(onset),

            "duration_sec":
                float(duration),

            "description":
                description,

            "condition":
                condition,

            "is_analysis_condition":
                condition is not None,
        })

    annotation_timeline_df = (
        pd.DataFrame(timeline_rows)
        .sort_values(
            [
                "onset_sec",
                "annotation_idx",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    state["annotation_timeline_df"] = (
        annotation_timeline_df.copy()
    )

    # ------------------------------------------------------------------
    # 4) Extract only the requested analysis-condition markers
    # ------------------------------------------------------------------
    condition_markers_df = (
        annotation_timeline_df.loc[
            annotation_timeline_df[
                "is_analysis_condition"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    n_condition_markers = len(
        condition_markers_df
    )

    if n_condition_markers < 2:
        return _cannot_calculate(
            "Fewer than two analysis-condition annotations were found.",
            timeline_df=annotation_timeline_df,
            summary_updates={
                "n_condition_markers":
                    int(n_condition_markers),
            },
        )

    observed_conditions = set(
        condition_markers_df[
            "condition"
        ].dropna()
    )

    missing_conditions = [
        condition
        for condition in condition_names
        if condition not in observed_conditions
    ]

    if missing_conditions:
        return _cannot_calculate(
            "Not all requested alternating conditions were observed. "
            f"Missing: {missing_conditions}",
            timeline_df=annotation_timeline_df,
            summary_updates={
                "n_condition_markers":
                    int(n_condition_markers),
            },
        )

    # ------------------------------------------------------------------
    # 5) Build candidate intervals from the condition annotations
    # ------------------------------------------------------------------
    #
    # Zero-duration annotations are treated as state markers:
    #
    #     current marker -> next condition marker
    #
    # For the final condition marker, the recording end provides the
    # candidate endpoint.
    #
    # Positive-duration condition annotations retain their explicit
    # annotation duration.
    # ------------------------------------------------------------------
    sfreq = float(
        raw.info["sfreq"]
    )

    recording_duration_sec = (
        float(raw.n_times)
        / sfreq
    )

    interval_rows = []

    for marker_position in range(
        n_condition_markers
    ):
        current = (
            condition_markers_df
            .iloc[marker_position]
        )

        current_onset = float(
            current["onset_sec"]
        )

        current_duration = float(
            current["duration_sec"]
        )

        current_condition = str(
            current["condition"]
        )

        current_annotation_idx = int(
            current["annotation_idx"]
        )

        has_next_condition = (
            marker_position
            < n_condition_markers - 1
        )

        if has_next_condition:
            next_marker = (
                condition_markers_df
                .iloc[marker_position + 1]
            )

            next_onset = float(
                next_marker["onset_sec"]
            )

            next_condition = str(
                next_marker["condition"]
            )

            next_annotation_idx = int(
                next_marker["annotation_idx"]
            )

        else:
            next_marker = None
            next_onset = np.nan
            next_condition = None
            next_annotation_idx = None

        # --------------------------------------------------------------
        # Determine candidate interval endpoint
        # --------------------------------------------------------------
        if current_duration > 0:
            interval_end = min(
                current_onset
                + current_duration,
                recording_duration_sec,
            )

            interval_source = (
                "annotation_duration"
            )

        elif has_next_condition:
            interval_end = (
                next_onset
            )

            interval_source = (
                "next_condition_marker"
            )

        else:
            interval_end = (
                recording_duration_sec
            )

            interval_source = (
                "recording_end"
            )

        interval_duration = (
            interval_end
            - current_onset
        )

        # --------------------------------------------------------------
        # Structural alternation check
        # --------------------------------------------------------------
        #
        # For an alternating two-condition design:
        #
        #     A -> B = structurally valid
        #     B -> A = structurally valid
        #
        #     A -> A = structurally ambiguous
        #     B -> B = structurally ambiguous
        #
        # The final marker has no following condition marker, so its
        # interval is evaluated later using timing rather than transition.
        # --------------------------------------------------------------
        if has_next_condition:
            alternation_valid = (
                current_condition
                != next_condition
            )
        else:
            alternation_valid = None

        # Explicit annotations should not overlap the next condition.
        overlaps_next_condition = False

        if (
            current_duration > 0
            and has_next_condition
            and interval_end > next_onset
        ):
            overlaps_next_condition = True

        # --------------------------------------------------------------
        # Preserve any non-condition annotations occurring inside the
        # candidate interval.
        #
        # These annotations are QC evidence only. They do NOT automatically
        # change or reject an eye state.
        # --------------------------------------------------------------
        intervening_annotations_df = (
            annotation_timeline_df.loc[
                (
                    annotation_timeline_df[
                        "onset_sec"
                    ] > current_onset
                )
                &
                (
                    annotation_timeline_df[
                        "onset_sec"
                    ] < interval_end
                )
                &
                (
                    ~annotation_timeline_df[
                        "is_analysis_condition"
                    ]
                )
            ]
        )

        intervening_annotations = (
            intervening_annotations_df[
                "description"
            ]
            .astype(str)
            .tolist()
        )

        interval_rows.append({
            "marker_position":
                int(marker_position),

            "annotation_idx":
                current_annotation_idx,

            "condition":
                current_condition,

            "onset_sec":
                current_onset,

            "end_sec":
                float(interval_end),

            "interval_duration_sec":
                float(interval_duration),

            "interval_source":
                interval_source,

            "has_next_condition":
                bool(has_next_condition),

            "next_annotation_idx":
                next_annotation_idx,

            "next_condition":
                next_condition,

            "alternation_valid":
                alternation_valid,

            "overlaps_next_condition":
                bool(overlaps_next_condition),

            "intervening_annotations":
                intervening_annotations,

            "n_intervening_annotations":
                len(intervening_annotations),
        })

    interval_qc_df = pd.DataFrame(
        interval_rows
    )

    # ------------------------------------------------------------------
    # 6) Learn the normal timing ONLY from structurally valid transitions
    # ------------------------------------------------------------------
    #
    # This is the central data-driven part of the method.
    #
    # Repeated condition markers are NOT used to define the normal timing.
    #
    # Therefore:
    #
    #     EC -> EC over 300 seconds
    #
    # cannot distort the recording-specific timing reference.
    # ------------------------------------------------------------------
    reference_mask = (
        interval_qc_df[
            "has_next_condition"
        ]
        &
        (
            interval_qc_df[
                "alternation_valid"
            ] == True
        )
        &
        (
            ~interval_qc_df[
                "overlaps_next_condition"
            ]
        )
        &
        (
            interval_qc_df[
                "interval_duration_sec"
            ] > 0
        )
    )

    reference_durations = (
        interval_qc_df.loc[
            reference_mask,
            "interval_duration_sec",
        ]
        .astype(float)
        .to_numpy()
    )

    n_reference_intervals = int(
        len(reference_durations)
    )

    if (
        n_reference_intervals
        < min_reference_intervals
    ):
        state[
            "annotation_validation_qc_df"
        ] = interval_qc_df.copy()

        return _cannot_calculate(
            "Insufficient valid alternating intervals to learn the "
            "recording-specific annotation timing.",
            timeline_df=annotation_timeline_df,
            qc_df=interval_qc_df,
            summary_updates={
                "n_condition_markers":
                    int(n_condition_markers),

                "n_reference_intervals":
                    int(n_reference_intervals),
            },
        )

    # Robust recording-specific center and spread.
    reference_median_sec = float(
        np.median(
            reference_durations
        )
    )

    reference_mad_sec = float(
        np.median(
            np.abs(
                reference_durations
                - reference_median_sec
            )
        )
    )

    # ------------------------------------------------------------------
    # 7) Evaluate the timing of every candidate interval
    # ------------------------------------------------------------------
    timing_modified_z = []
    timing_valid = []

    for interval_duration in (
        interval_qc_df[
            "interval_duration_sec"
        ].astype(float)
    ):

        if (
            not np.isfinite(
                interval_duration
            )
            or interval_duration <= 0
        ):
            timing_modified_z.append(
                np.nan
            )

            timing_valid.append(
                False
            )

            continue

        if reference_mad_sec > 0:
            modified_z = (
                0.67448975
                * (
                    interval_duration
                    - reference_median_sec
                )
                / reference_mad_sec
            )

            is_timing_valid = (
                abs(modified_z)
                <= timing_modified_z_threshold
            )

        else:
            # ----------------------------------------------------------
            # MAD can be zero when most valid intervals have exactly
            # the same duration.
            #
            # In that case, use acquisition precision itself as the
            # numerical tolerance rather than inventing a duration-based
            # threshold.
            # ----------------------------------------------------------
            sample_duration_sec = (
                1.0 / sfreq
            )

            difference_sec = abs(
                interval_duration
                - reference_median_sec
            )

            is_timing_valid = (
                difference_sec
                <= sample_duration_sec
            )

            modified_z = (
                0.0
                if is_timing_valid
                else np.inf
            )

        timing_modified_z.append(
            float(modified_z)
        )

        timing_valid.append(
            bool(is_timing_valid)
        )

    interval_qc_df[
        "timing_modified_z"
    ] = timing_modified_z

    interval_qc_df[
        "timing_valid"
    ] = timing_valid

    # ------------------------------------------------------------------
    # 8) Make the final interval decision
    # ------------------------------------------------------------------
    accepted = []
    decision_reason = []

    for row in (
        interval_qc_df
        .itertuples(index=False)
    ):

        # Invalid or empty interval
        if (
            not np.isfinite(
                row.interval_duration_sec
            )
            or row.interval_duration_sec <= 0
        ):
            accepted.append(
                False
            )

            decision_reason.append(
                "invalid_interval_duration"
            )

            continue

        # Explicit annotation overlaps a following condition marker
        if row.overlaps_next_condition:
            accepted.append(
                False
            )

            decision_reason.append(
                "condition_interval_overlaps_next_marker"
            )

            continue

        # Consecutive identical condition states violate alternation
        if (
            row.has_next_condition
            and row.alternation_valid is False
        ):
            accepted.append(
                False
            )

            decision_reason.append(
                "repeated_condition_marker"
            )

            continue

        # Timing inconsistent with the valid alternating intervals
        if not row.timing_valid:
            accepted.append(
                False
            )

            decision_reason.append(
                "timing_outlier"
            )

            continue

        # Remaining intervals are interpretable.
        #
        # This includes the terminal interval because its duration can
        # still be compared with the learned recording-specific timing.
        accepted.append(
            True
        )

        if row.has_next_condition:
            decision_reason.append(
                "accepted_alternating_interval"
            )
        else:
            decision_reason.append(
                "accepted_terminal_interval"
            )

    interval_qc_df[
        "accepted"
    ] = accepted

    interval_qc_df[
        "decision_reason"
    ] = decision_reason

    # ------------------------------------------------------------------
    # 9) Build the validated condition intervals
    # ------------------------------------------------------------------
    validated_condition_intervals = {
        condition: []
        for condition in condition_names
    }

    accepted_df = (
        interval_qc_df.loc[
            interval_qc_df[
                "accepted"
            ]
        ]
        .copy()
    )

    for row in (
        accepted_df
        .itertuples(index=False)
    ):
        validated_condition_intervals[
            row.condition
        ].append({
            "onset":
                float(row.onset_sec),

            "duration":
                float(
                    row.interval_duration_sec
                ),

            "end":
                float(row.end_sec),

            "source_annotation_idx":
                int(row.annotation_idx),

            "interval_source":
                str(row.interval_source),
        })

    # ------------------------------------------------------------------
    # 10) Make sure both alternating conditions remain analyzable
    # ------------------------------------------------------------------
    conditions_without_intervals = [
        condition
        for condition in condition_names
        if not validated_condition_intervals[
            condition
        ]
    ]

    if conditions_without_intervals:
        state[
            "annotation_validation_qc_df"
        ] = interval_qc_df.copy()

        return _cannot_calculate(
            "Annotation validation did not produce at least one "
            "confident interval for every requested condition. "
            f"No accepted intervals for: {conditions_without_intervals}",
            timeline_df=annotation_timeline_df,
            qc_df=interval_qc_df,
            summary_updates={
                "n_condition_markers":
                    int(n_condition_markers),

                "n_reference_intervals":
                    int(n_reference_intervals),

                "reference_median_sec":
                    reference_median_sec,

                "reference_mad_sec":
                    reference_mad_sec,
            },
        )

    # ------------------------------------------------------------------
    # 11) Build successful recording-level QC summary
    # ------------------------------------------------------------------
    accepted_duration_by_condition = {
        condition: float(
            accepted_df.loc[
                accepted_df[
                    "condition"
                ] == condition,
                "interval_duration_sec",
            ].sum()
        )
        for condition in condition_names
    }

    accepted_interval_count_by_condition = {
        condition: int(
            (
                accepted_df[
                    "condition"
                ]
                == condition
            ).sum()
        )
        for condition in condition_names
    }

    n_accepted_intervals = int(
        interval_qc_df[
            "accepted"
        ].sum()
    )

    n_rejected_intervals = int(
        len(interval_qc_df)
        - n_accepted_intervals
    )

    n_repeated_condition_intervals = int(
        (
            interval_qc_df[
                "decision_reason"
            ]
            == "repeated_condition_marker"
        ).sum()
    )

    n_timing_outliers = int(
        (
            interval_qc_df[
                "decision_reason"
            ]
            == "timing_outlier"
        ).sum()
    )

    annotation_validation_summary = {
        "status":
            "ok",

        "error":
            None,

        "conditions":
            list(condition_names),

        "n_total_annotations":
            int(
                len(
                    annotation_timeline_df
                )
            ),

        "n_condition_markers":
            int(n_condition_markers),

        "n_reference_intervals":
            int(n_reference_intervals),

        "timing_method":
            "modified_z_from_median_mad",

        "timing_modified_z_threshold":
            float(
                timing_modified_z_threshold
            ),

        "reference_median_sec":
            reference_median_sec,

        "reference_mad_sec":
            reference_mad_sec,

        "n_candidate_intervals":
            int(
                len(
                    interval_qc_df
                )
            ),

        "n_accepted_intervals":
            n_accepted_intervals,

        "n_rejected_intervals":
            n_rejected_intervals,

        "n_repeated_condition_intervals":
            n_repeated_condition_intervals,

        "n_timing_outliers":
            n_timing_outliers,

        "accepted_interval_count_by_condition":
            accepted_interval_count_by_condition,

        "accepted_duration_seconds_by_condition":
            accepted_duration_by_condition,
    }

    # ------------------------------------------------------------------
    # 12) Save outputs to pipeline state
    # ------------------------------------------------------------------
    state[
        "annotation_validation_status"
    ] = "ok"

    state[
        "annotation_validation_error"
    ] = None

    state[
        "annotation_timeline_df"
    ] = annotation_timeline_df

    state[
        "annotation_validation_qc_df"
    ] = interval_qc_df

    state[
        "annotation_validation_summary"
    ] = annotation_validation_summary

    state[
        "validated_condition_intervals"
    ] = validated_condition_intervals

    # ------------------------------------------------------------------
    # 13) Optional concise reporting
    # ------------------------------------------------------------------
    if verbose:
        print(
            "[annotation validation] "
            "Status: OK"
        )

        print(
            "[annotation validation] "
            f"Conditions: {condition_names}"
        )

        print(
            "[annotation validation] "
            f"Condition markers: {n_condition_markers}"
        )

        print(
            "[annotation validation] "
            f"Reference intervals: {n_reference_intervals}"
        )

        print(
            "[annotation validation] "
            f"Learned median interval: "
            f"{reference_median_sec:.3f} sec"
        )

        print(
            "[annotation validation] "
            f"Learned MAD: "
            f"{reference_mad_sec:.3f} sec"
        )

        print(
            "[annotation validation] "
            f"Accepted intervals: "
            f"{n_accepted_intervals}"
        )

        print(
            "[annotation validation] "
            f"Rejected intervals: "
            f"{n_rejected_intervals}"
        )

        print(
            "[annotation validation] "
            "Accepted duration by condition: "
            f"{accepted_duration_by_condition}"
        )

    return state


def _build_ops() -> Dict[str, StepFn]:
    # Maps step names used in config -> the function that executes that step
    return {
        "load_eeg": step_load_eeg,                      # Load EEG file into state["raw"]
        "detect_analysis_conditions": step_detect_analysis_conditions,  #  Detect analysis conditions from Raw annotations and reconstruct their intervals
        "mark_non_analysis_segments": step_mark_non_analysis_segments,  # Mark portions outside recognized analysis conditions as BAD_non_analysis.
        "scale_data": step_scale_data,                  # Scale EEG data from uV to V
        "auto_scale_data": step_auto_scale_data,        # New automatic EEG scaling data
        "set_montage": step_set_montage,                # Attach electrode positions (montage) to raw
        "drop_channels": step_drop_channels,            # Remove unwanted channels (e.g., M1/M2)
        "demote_unlocalized": step_demote_unlocalized,  # Convert EEG chans with missing xyz to 'misc'
        "resample_eeg": step_resample_eeg,              # Change sampling rate (updates raw.info["sfreq"])
        "bandpass_filter": step_bandpass,               # Band-pass filter raw (e.g., 0.5–45 Hz)
        "notch_filter": step_notch_filter,              # Notch filter line noise (e.g., 60/120 Hz)
        "plot_raw": step_plot_raw,                      # Plot time-series browser (raw.plot)
        "plot_psd": step_plot_psd,                      # Plot PSD (raw.compute_psd(...).plot())
        "mad_bad_channels": step_mad_bad_channels,          # MAD amplitude-outlier detection + interpolation
        "ransac_clean": step_ransac_clean,              # RANSAC bad-channel detection + interpolation
        "set_reference": step_set_reference,            # Re-reference EEG (e.g., average reference)
        "apply_csd": step_apply_csd,                    # Apply CSD / surface Laplacian (spatial sharpening)
        "fixed_length_epochs": step_fixed_length_epochs, # Create fixed-length events + epochs (final segments)
        "reject_bad_epochs": step_reject_bad_epochs,    # Reject bad epochs
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



config_nehemiah  = {
    "steps": [
        # 1) Load EEG file
        {"load_eeg": {"params": {"path": "EEG1.bdf"}, "verbose": True}},

        # 2) Assign scalp electrode positions
        {"set_montage": {"params": {"kind": "biosemi64"}, "verbose": True}},

        # 3) Resample early (affects Nyquist + speeds later filtering)
        {"resample_eeg": {"params": {"sfreq": 250.0,  "verbose": True}, "verbose": True}},

        # 4) Quick QC: raw view
        {"plot_raw": {"params": {"n_channels": 129, "picks": "eeg"}, "verbose": True}},

        # 5) Quick QC: PSD
        {"plot_psd": {"params": {"picks": "eeg", "average": False}, "verbose": True}},

        # 6) Drop mastoids early
        {"drop_channels": {"params": {"names": ["M1", "M2"]}, "verbose": True}},

        # 7) Mark channels without valid xyz as 'misc' (prevents spatial ops issues)
        {"demote_unlocalized": {"params": {}, "verbose": True}},

        # 8) Notch filter (line noise) — ensure freqs < Nyquist
        {"notch_filter": {"params": {"freqs": [60.0, ], # 120.0 
                                     "phase": "zero", "filter_length": "auto", "verbose": True}, "verbose": True}},

        # 9) Band-pass filter
        {"bandpass_filter": {"params": {"l_freq": 0.5, "h_freq": 45.0, "phase": "zero", "fir_design": "firwin",  "verbose": False}, "verbose": True}},

        # 10) QC again: raw
        {"plot_raw": {"params": {"n_channels": 129, "picks": "eeg"}, "verbose": True}},

        # 11) QC again: PSD
        {"plot_psd": {"params": {"picks": "eeg", "average": False}, "verbose": True}},

        # 12) RANSAC bad-channel detection + interpolation
        {"ransac_clean": {"params": {
            "events": {"duration": 2.0, "overlap": 0.0, "id": 2,},
            "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, "baseline": None,
                       "reject": None, "detrend": 0, "preload": True, "verbose": False},
            "ransac": {"n_jobs": -1, "verbose": False},
            "reset_bads": True
        }, "verbose": True}},

        # 13) Average reference
        {"set_reference": {"params": {"ref_channels": "average", "verbose": False}, "verbose": True}},

        # 14) ICA + ICLabel artifact-component removal
        {"run_ica_iclabel": {"params": {
            "ica": {
                "n_components": 0.99,
                "method": "infomax",
                "fit_params": {"extended": True},
                "random_state": 42,
                "max_iter": "auto"
            },
            "fit": {
                "notch_freqs": None,
                "l_freq": 1.0,
                "desired_h_freq": 100.0,
                "apply_average_ref": False,
                "picks": "eeg"
            },
            "iclabel": {
                "artifact_labels": [
                    "eye blink",
                    "muscle artifact",
                    "line noise",
                    "heart beat",
                    "channel noise"
                ],
                "prob_threshold": 0.8
            }
        }, "verbose": True}},

        # 15) CSD (surface Laplacian)
        {"apply_csd": {"params": {"verbose": False}, "verbose": True}},

        # 16) Final fixed-length epochs for ML
        {"fixed_length_epochs": {"params": {
            "events": {"duration": 2.0, "overlap": 0.0, "id": 2},
            "epochs": {"event_id": {"2s_segment": 2}, "tmin": 0.0, "tmax": 2.0, "baseline": None,
                       "reject": None, "detrend": 0, "preload": True, "verbose": False},
            "store": {"events_key": "events_final", "epochs_key": "epochs_final"}
        }, "verbose": False}},

        # 17) Final PSD check (CSD’d data). Note "picks": "data" after doing PSD since 'eeg' gets replaced
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

    