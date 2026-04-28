
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microstate EEG Pipeline (single-file, **no preprocessing**)

Call these in your notebook:
    1) raw_clean, epochs_final, info = preprocess_eeg(config)   # from your own module
    2) results = pipeline_microstates(epochs_final, picks="eeg", ...)

This module performs:
 - GFP peak extraction
 - K-grid fit with ModKMeans
 - Elbow-based K selection
 - Final fit + backfitting
 - (optional) elbow plot

Returns a PipelineResults dataclass with df_k, models, best_k, modk, seg, labels, fs, k, state_names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from tqdm.auto import tqdm 

from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.cluster import ModKMeans

def plot_k_elbow(df, best_k, min_gain_pct=10.0):
    """
    Plot the total Global Explained Variance (GEV) and its percent gain (ΔGEV%)
    across different numbers of microstates (K), helping to identify the optimal K.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `fit_k_grid()`. Must contain the following columns:
        - 'K' : integer number of clusters tested
        - 'total_gev' : total Global Explained Variance for each K
        - 'delta_gev_pct' : percent gain in GEV from K-1 to K
    best_k : int
        The chosen K (e.g., selected using `choose_k_by_elbow`).
    min_gain_pct : float, optional
        Horizontal cutoff (in percent) marking the minimum acceptable GEV gain.
        Default = 10.0 (%). Used to visually indicate the practical elbow criterion.

    Notes
    -----
    **What the plot shows**
    - **Left axis (blue line)**: Total GEV — how much EEG variance is explained by
      the model for each number of microstates. It always increases with K.
    - **Right axis (orange dashed line)**: Percent gain in GEV (ΔGEV%) — the gain
      you get when adding another cluster.
    - **Horizontal dashed line (orange)**: The cutoff threshold (e.g., 10%) used
      to mark diminishing returns in GEV improvement.
    - **Vertical dotted line (black)**: The chosen K and its total GEV (red marker).

    **How to interpret**
    - The optimal K is typically where the orange curve (ΔGEV%) drops *below*
      the horizontal cutoff line (e.g., <10%) or where the orange curve flattens.
    - Beyond this point, adding more clusters explains little new variance and
      can make microstate maps redundant or noisy.
    - The total GEV curve (blue) should rise quickly at first and then plateau.
    - Typical values:
        - Resting-state EEG: K ≈ 4 ± 1
        - Task-related or shorter data: K ≈ 4–6
    - Always confirm visually that the microstate maps for chosen K are distinct
      and interpretable.

    Returns
    -------
    None
        Displays the GEV elbow plot using matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["K"], df["total_gev"], "o-", label="Total GEV")
    ax1.set_xlabel("Number of microstates (K)")
    ax1.set_ylabel("Total GEV", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Secondary axis for % gain
    ax2 = ax1.twinx()
    ax2.plot(df["K"], df["delta_gev_pct"], "s--", color="C1", label="ΔGEV (%)")
    ax2.set_ylabel("Percent gain in total GEV", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    # --- Add horizontal cutoff line ---
    ax2.axhline(min_gain_pct, color="black", linestyle="--", alpha=0.6)
    # Position label horizontally in the middle of the x-axis
    mid_x = (df["K"].min() + df["K"].max()) / 2
    ax2.text(
        mid_x,
        min_gain_pct,
        f"{min_gain_pct:.0f}% cutoff",
        color="black",
        va="bottom",
        ha="center",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
    )

    # Highlight the chosen K
    best_row = df[df["K"] == best_k].iloc[0]
    ax1.axvline(best_k, color="k", linestyle=":", alpha=0.7)
    ax1.scatter(best_k, best_row["total_gev"], color="red", s=80, zorder=5)
    ax1.text(best_k + 0.1, best_row["total_gev"],
             f"K={best_k}", color="red", va="bottom", fontsize=10)

    plt.title("Choosing optimal K by GEV elbow")
    fig.tight_layout()
    plt.show()



def fit_k_grid(
    epochs,
    gfp_epochs,
    k_range=range(2, 8),
    picks="eeg",
    n_init=100,
    max_iter=300,
    random_state=42,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
    reject_by_annotation=True,
    factor=10,
    show_progress=True,
):
    """
    Fit ModKMeans microstate models for multiple cluster numbers (K)
    and compute summary metrics to help select the optimal K.

    Parameters
    ----------
    epochs : mne.Epochs
        Full epoched EEG data used for segmentation (backfitting).
    gfp_epochs : mne.Epochs
        GFP-peak–sampled data used for model fitting.
    k_range : range, optional
        Range of K values (number of microstates) to evaluate. Default = range(2, 8).
    picks : str or list, optional
        Channels to include (e.g., "eeg"). Default = "eeg".
    n_init : int, optional
        Number of K-means initializations for stability. Default = 100.
    max_iter : int, optional
        Maximum number of iterations for the ModKMeans algorithm. Default = 300.
    random_state : int, optional
        Seed for reproducibility. Default = 42.
    half_window_size : int, optional
        Temporal smoothing half-window in samples. Default = 10.
    min_segment_length : int, optional
        Minimum allowed microstate segment length (samples). Default = 5.
    reject_edges : bool, optional
        Whether to ignore epoch edges when computing durations. Default = True.
    reject_by_annotation : bool, optional
        Whether to reject annotated (bad) segments. Default = True.
    factor : float, optional
        Temporal smoothing factor. Larger = smoother segmentation. Default = 10.
    show_progress : bool, optional
        If True, show a tqdm progress bar. Default = True.

    Returns
    -------
    df : pandas.DataFrame
        Summary table of total GEV, mean duration, and time coverage per K.
    models : dict
        Dictionary with fitted models and segmentation results for each K.
    """
    results = []
    models = {}

    # Wrap k_range in tqdm if progress display is enabled
    iterator = tqdm(k_range, desc="Fitting ModKMeans (K-grid)", unit="K") if show_progress else k_range

    for k in iterator:
        # Optional: update tqdm text instead of printing
        if not show_progress:
            print(f"Fitting ModKMeans for K={k} ...")

        # --- Fit ModKMeans model on GFP peaks ---
        mk = ModKMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        mk.fit(gfp_epochs, picks=picks, n_jobs=10)

        # --- Backfit model to full epochs ---
        seg = mk.predict(
            epochs,
            picks=picks,
            reject_by_annotation=reject_by_annotation,
            factor=factor,
            half_window_size=half_window_size,
            min_segment_length=min_segment_length,
            reject_edges=reject_edges,
        )

        # --- Compute metrics ---
        params = seg.compute_parameters(norm_gfp=True)

        gev_vals = [v for k_, v in params.items() if k_.endswith("_gev")]
        dur_vals = [v for k_, v in params.items() if k_.endswith("_meandurs")]
        cov_vals = [v for k_, v in params.items() if k_.endswith("_timecov")]

        total_gev = float(np.sum(gev_vals))
        mean_duration = float(np.mean(dur_vals))
        mean_timecov = float(np.mean(cov_vals))

        results.append({
            "K": k,
            "total_gev": total_gev,
            "mean_duration_ms": mean_duration,
            "mean_time_coverage": mean_timecov,
        })

        models[k] = {"model": mk, "seg": seg, "params": params}

    # --- Combine results into summary DataFrame ---
    df = pd.DataFrame(results).sort_values("K").reset_index(drop=True)
    df["delta_gev_pct"] = df["total_gev"].pct_change() * 100.0

    return df, models

def choose_k_by_elbow(df, min_gain_pct=5.0):
    """
    Simple elbow: choose first K where % gain in total GEV drops below min_gain_pct.
    Falls back to max GEV if no elbow found.
    """
    for i in range(1, len(df)):
        if df.loc[i, "delta_gev_pct"] < min_gain_pct:
            return int(df.loc[i, "K"])
    return int(df.loc[df["total_gev"].idxmax(), "K"])




def fit_final_microstates(
    epochs,
    gfp_epochs,
    best_k,
    picks="eeg",
    n_jobs=10,
    n_init=200,
    max_iter=500,
    random_state=42,
    factor=10,
    half_window_size=10,
    min_segment_length=5,
    reject_edges=True,
    reject_by_annotation=True,
):
    """
    Fit the final ModKMeans microstate model using the chosen K value,
    backfit it to the full epochs, and return key outputs for downstream analysis.

    Parameters
    ----------
    epochs : mne.Epochs
        Full EEG data for segmentation (backfitting).
    gfp_epochs : mne.Epochs
        GFP-peak sampled epochs used for clustering.
    best_k : int
        Optimal number of microstates (from GEV elbow selection).
    picks : str or list, optional
        Channel selection for clustering (e.g., "eeg" or "data"). Default = "eeg".
    n_jobs : int, optional
        Number of CPU cores to use for parallelization. Default = 10.
    n_init : int, optional
        Number of K-means initializations for stability. Default = 200.
    max_iter : int, optional
        Maximum number of iterations for ModKMeans. Default = 500.
    random_state : int, optional
        Seed for reproducibility. Default = 42.
    factor : float, optional
        Temporal smoothing factor. Larger = smoother segmentation. Default = 10.
    half_window_size : int, optional
        Temporal smoothing half-window (in samples). Default = 10.
    min_segment_length : int, optional
        Minimum allowed microstate segment length (samples). Default = 5.
    reject_edges : bool, optional
        Whether to ignore epoch edges when computing durations. Default = True.
    reject_by_annotation : bool, optional
        Whether to exclude annotated (bad) segments. Default = True.

    Returns
    -------
    modk : pycrostates.cluster.ModKMeans
        The fitted ModKMeans model.
    seg : pycrostates.segmentation.EpochsSegmentation
        Segmentation object containing microstate labels and maps.
    labels : ndarray, shape (n_epochs, n_times)
        Microstate label assignments for each sample.
    fs : float
        Sampling frequency of the input data.
    k : int
        Number of identified microstates (equals best_k).
    state_names : list of str
        Alphabetical state names, e.g., ['A', 'B', 'C', 'D'].
    """
    # --- Fit final ModKMeans model ---
    modk = ModKMeans(
        n_clusters=int(best_k),
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    modk.fit(gfp_epochs, picks=picks, n_jobs=n_jobs)

    # --- Backfit (segment) the full dataset ---
    seg = modk.predict(
        epochs,
        picks=picks,
        reject_by_annotation=reject_by_annotation,
        factor=factor,
        half_window_size=half_window_size,
        min_segment_length=min_segment_length,
        reject_edges=reject_edges,
    )

    # --- Prepare outputs ---
    labels = seg.labels                   # (n_epochs, n_times)
    fs = float(epochs.info["sfreq"])      # sampling frequency
    k = len(modk.cluster_names)           # number of clusters
    state_names = [chr(ord("A") + i) for i in range(k)]  # ['A', 'B', 'C', ...]

    print(f"Final model fitted with K={k} microstates ({state_names}), fs={fs} Hz")

    return modk, seg, labels, fs, k, state_names


def pipeline_microstates_multi_subject(
    label_epoch_dict: Dict[str, List["mne.Epochs"]],
    metadata: List[Dict[str, str]],
    picks: str = "eeg",
    k_range=range(2, 10),
    min_gain_pct: float = 10.0,
    min_peak_distance_s: float = 0.003,
    show_progress: bool = True,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    Run microstates per subject given a dict[label -> list[mne.Epochs]] and matching metadata.

    Returns
    -------
    {
      "summary_df": pd.DataFrame  # one row per subject
      "artifacts": { (label, subject_id): {
           "labels": np.ndarray (n_epochs, n_times),
           "k": int,
           "state_names": list[str],
           "fs": float,
           "df_k": pd.DataFrame
        }, ... },
      "logs": list[dict]          # 'ok' or 'skipped' per subject
    }
    """
    if not label_epoch_dict or not metadata:
        raise ValueError("Provide non-empty label_epoch_dict and metadata.")

    # Iterate deterministically by label, then subject
    meta_sorted = sorted(metadata, key=lambda r: (r["label"], r["subject_id"]))
    label_counters = {lab: 0 for lab in label_epoch_dict.keys()}

    summary_rows: List[Dict[str, Any]] = []
    artifacts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    logs: List[Dict[str, str]] = []

    for m in meta_sorted:
        lab = m["label"]
        sid = m["subject_id"]

        # Match the next Epochs for this label
        i = label_counters.get(lab, 0)
        if lab not in label_epoch_dict or i >= len(label_epoch_dict[lab]):
            logs.append({"label": lab, "subject_id": sid, "status": "skipped", "reason": "no Epochs found"})
            continue

        ep = label_epoch_dict[lab][i].copy()
        label_counters[lab] = i + 1

        try:
            fs = float(ep.info["sfreq"])
            min_peak_distance = max(1, int(round(min_peak_distance_s * fs)))

            # 1) GFP peaks
            gfp_epochs = extract_gfp_peaks(ep, picks=picks, min_peak_distance=min_peak_distance)

            # 2) K-grid → choose K
            df_k, _models = fit_k_grid(
                epochs=ep,
                gfp_epochs=gfp_epochs,
                k_range=k_range,
                picks=picks,
                show_progress=show_progress,
            )
            best_k = int(choose_k_by_elbow(df_k, min_gain_pct=min_gain_pct))

            if plot:
                plot_k_elbow(df_k, best_k, min_gain_pct=min_gain_pct)

            # 3) Final fit
            modk, seg, labels, fs_final, k_final, state_names = fit_final_microstates(
                epochs=ep,
                gfp_epochs=gfp_epochs,
                best_k=best_k,
                picks=picks,
                n_jobs=10,
            )

            # Pull summary metrics from the chosen-K row
            row_k = df_k.loc[df_k["K"] == best_k].iloc[0]
            summary_rows.append({
                "label": lab,
                "subject_id": sid,
                "best_k": best_k,
                "total_gev": float(row_k["total_gev"]),
                "mean_duration_ms": float(row_k["mean_duration_ms"]),
                "mean_time_coverage": float(row_k["mean_time_coverage"]),
                "sfreq": fs_final,
                "n_epochs": len(ep),
            })

            # Keep compact artifacts; models are ephemeral
            artifacts[(lab, sid)] = {
                "labels": labels,                # (n_epochs, n_times)
                "k": int(k_final),
                "state_names": list(state_names),
                "fs": float(fs_final),
                "df_k": df_k.copy(),
            }

            logs.append({"label": lab, "subject_id": sid, "status": "ok"})

            # Drop heavy objects before next subject
            del modk, seg, gfp_epochs, _models

        except Exception as e:
            logs.append({"label": lab, "subject_id": sid, "status": "skipped", "reason": f"{e}"})
            continue

    summary_df = pd.DataFrame(summary_rows).sort_values(["label", "subject_id"]).reset_index(drop=True)
    return {"summary_df": summary_df, "artifacts": artifacts, "logs": logs}
