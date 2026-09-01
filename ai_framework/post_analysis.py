# post_analysis.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union, Sequence, List, Mapping, Literal
import math
from statistics import NormalDist
from scipy.stats import binom, binomtest
import matplotlib.pyplot as plt
import seaborn as sns

Threshold = Union[float, Tuple[float, float]]
Stratum = Tuple[str, float, float]





def preprocess_by_threshold(
    df: pd.DataFrame,
    threshold: Threshold,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    models: Optional[Union[str, Sequence[str]]] = None,
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    keep_cols: Optional[List[str]] = None,
    enforce_unique: bool = True,
) -> pd.DataFrame:
    """
    Return an analysis-ready subject-level dataframe filtered by:
      - single threshold: score >= threshold
      - interval/band:     low <= score < high   (always half-open: [low, high))

    This makes bands non-overlapping by construction.
    """
    if grouping_keys is None:
        grouping_keys = ["model", "variant", "split", "grouping", "unit_col"]

    required = set(grouping_keys + ["subject_id", "y", "group_label", score_col])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Filter population
    if split is not None:
        out = out[out["split"] == split]

    if models is not None:
        if isinstance(models, str):
            models = [models]
        out = out[out["model"].isin(list(models))]

    if variants is not None:
        if isinstance(variants, str):
            variants = [variants]
        out = out[out["variant"].isin(list(variants))]

    # Drop rows without a usable score
    out = out.dropna(subset=[score_col])

    # Enforce one row per subject per evaluation context (optional)
    key = grouping_keys + ["subject_id"]
    if enforce_unique:
        dup_mask = out.duplicated(subset=key, keep=False)
        if dup_mask.any():
            # Prefer the row with the most windows; tie-break by higher score
            sort_cols = []
            if "n_windows" in out.columns:
                sort_cols.append("n_windows")
            sort_cols.append(score_col)

            out = (
                out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                   .drop_duplicates(subset=key, keep="first")
            )

    # Apply threshold selection
    s = out[score_col]
    if isinstance(threshold, tuple):
        low, high = threshold
        if low > high:
            raise ValueError(f"Invalid threshold interval: low ({low}) > high ({high})")
        out = out[(s >= low) & (s < high)].copy()
    else:
        out = out[s >= float(threshold)].copy()

    # Keep a small, useful set of columns by default
    if keep_cols is None:
        base = grouping_keys + ["subject_id", "group_label", "y", score_col]
        optional = [c for c in ["n_windows", "p_total_std", "lower_q", "upper_q"] if c in out.columns]
        keep_cols = base + optional

    keep_cols = [c for c in keep_cols if c in out.columns]
    return out.loc[:, keep_cols]

def _infer_label_map(
    df: pd.DataFrame,
    y_col: str = "y",
    label_col: str = "group_label",
) -> Dict[int, str]:
    """
    Infer mapping y -> label name using the most common label per y value.
    Falls back to string of y if label_col is missing/unusable.
    """
    if label_col not in df.columns:
        return {}

    sub = df[[y_col, label_col]].dropna()
    if sub.empty:
        return {}

    mapping: Dict[int, str] = {}
    for y_val in sorted(sub[y_col].astype(int).unique()):
        labels = sub.loc[sub[y_col].astype(int) == y_val, label_col].astype(str)
        if labels.empty:
            continue
        # mode() can return multiple values in ties; pick first for stability
        mapping[int(y_val)] = labels.mode().iloc[0]
    return mapping


def local_pocket_metrics(
    df_hi: pd.DataFrame,
    df_all: pd.DataFrame,
    y_col: str = "y",
    label_col: str = "group_label",
) -> Dict[str, Any]:
    """
    Compute local pocket metrics from threshold-selected df_hi, plus baseline prevalence from df_all.
    Assumes y=1 is the positive class.
    """
    if df_hi is None or df_hi.empty:
        raise ValueError("df_hi is empty. Did your threshold return any rows?")
    if df_all is None or df_all.empty:
        raise ValueError("df_all is empty. Provide the matching full evaluation dataframe.")

    for dname, d in [("df_hi", df_hi), ("df_all", df_all)]:
        if y_col not in d.columns:
            raise ValueError(f"{dname} is missing required column: {y_col}")

    # Local counts (in the pocket)
    y_hi = df_hi[y_col].dropna().astype(int)
    n_sel = int(len(y_hi))
    n_pos_sel = int((y_hi == 1).sum())
    n_neg_sel = int((y_hi == 0).sum())

    ppv = n_pos_sel / n_sel
    fdr = n_neg_sel / n_sel

    pos_neg_ratio = (n_pos_sel / n_neg_sel) if n_neg_sel > 0 else float("inf")

    # Baseline prevalence (in the full population)
    y_all = df_all[y_col].dropna().astype(int)
    n_all = int(len(y_all))
    n_pos_all = int((y_all == 1).sum())
    baseline_prev = n_pos_all / n_all if n_all > 0 else float("nan")

    enrichment = (ppv / baseline_prev) if baseline_prev and baseline_prev > 0 else float("inf")

    # Optional label counts
    label_counts_hi = None
    if label_col in df_hi.columns:
        label_counts_hi = df_hi[label_col].value_counts(dropna=False).to_dict()

    return {
        "n_selected": n_sel,
        "n_pos_selected": n_pos_sel,
        "n_neg_selected": n_neg_sel,
        "ppv_purity": ppv,
        "fdr_contamination": fdr,
        "pos_to_neg_ratio": pos_neg_ratio,
        "baseline_prevalence": baseline_prev,
        "enrichment_factor": enrichment,
        "label_counts_selected": label_counts_hi,
        # convenience: local prevalence in pocket (same as PPV here)
        "pocket_prevalence": ppv,
    }


def pocket_metrics_df(
    df_hi: pd.DataFrame,
    df_all: pd.DataFrame,
    threshold: Threshold,
    y_col: str = "y",
    label_col: str = "group_label",
    score_col: str = "p_mean",
    meta_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    1-row dataframe summarizing a threshold-defined pocket.

    Includes:
      - pocket_prevalence (same as PPV here)
      - baseline_prevalence + enrichment_factor
      - n_total + pct_selected
    """
    if meta_cols is None:
        meta_cols = ["model", "variant", "split"]

    m = local_pocket_metrics(df_hi=df_hi, df_all=df_all, y_col=y_col, label_col=label_col)

    # threshold reporting: [low, high)
    if isinstance(threshold, tuple):
        thr_low, thr_high = float(threshold[0]), float(threshold[1])
    else:
        thr_low, thr_high = float(threshold), 1.0

    # Infer label names for y=1 and y=0
    label_map = _infer_label_map(df_all, y_col=y_col, label_col=label_col)
    pos_label = label_map.get(1, "pos")
    neg_label = label_map.get(0, "neg")

    # metadata (if df_hi contains multiple values, keep list)
    meta: Dict[str, Any] = {}
    for c in meta_cols:
        if c in df_hi.columns:
            vals = df_hi[c].dropna().unique()
            meta[c] = vals[0] if len(vals) == 1 else list(vals)

    # Define n_total using the same cols you care about (y + score)
    n_total = int(len(df_all.dropna(subset=[y_col, score_col])))

    row = {
        **meta,
        "score_col": score_col,
        "thr_low": thr_low,
        "thr_high": thr_high,
        "pos_label": pos_label,
        "neg_label": neg_label,

        "n_selected": m["n_selected"],
        "n_total": n_total,
        "pct_selected": (m["n_selected"] / n_total) if n_total > 0 else float("nan"),

        "n_pos_selected": m["n_pos_selected"],
        "n_neg_selected": m["n_neg_selected"],

        "ppv": m["ppv_purity"],
        "fdr": m["fdr_contamination"],
        "pos_to_neg_ratio": m["pos_to_neg_ratio"],
        "baseline_prevalence": m["baseline_prevalence"],
        "enrichment_factor": m["enrichment_factor"],
    }

    return pd.DataFrame([row])

def preprocess_by_threshold(
    df: pd.DataFrame,
    threshold: Threshold,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    models: Optional[Union[str, Sequence[str]]] = None,
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    keep_cols: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Return an analysis-ready subject-level dataframe filtered by:
      - single threshold: score >= threshold
      - interval/band:     low <= score < high   (half-open: [low, high))

    Also supports dropping known-bad subjects via drop_subject_ids.
    """
    if grouping_keys is None:
        grouping_keys = ["model", "variant", "split", "grouping", "unit_col"]

    required = set(grouping_keys + [subject_col, "y", "group_label", score_col])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Drop subject IDs (QA exclusions)
    if drop_subject_ids is not None and len(drop_subject_ids) > 0:
        out = out[~out[subject_col].isin(list(drop_subject_ids))]

    # Filter population
    if split is not None:
        out = out[out["split"] == split]

    if models is not None:
        if isinstance(models, str):
            models = [models]
        out = out[out["model"].isin(list(models))]

    if variants is not None:
        if isinstance(variants, str):
            variants = [variants]
        out = out[out["variant"].isin(list(variants))]

    # Drop rows without a usable score
    out = out.dropna(subset=[score_col])

    # Enforce one row per subject per evaluation context (optional)
    key = grouping_keys + [subject_col]
    if enforce_unique:
        dup_mask = out.duplicated(subset=key, keep=False)
        if dup_mask.any():
            sort_cols = []
            if "n_windows" in out.columns:
                sort_cols.append("n_windows")
            sort_cols.append(score_col)

            out = (
                out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                   .drop_duplicates(subset=key, keep="first")
            )

    # Apply threshold selection
    s = out[score_col]
    if isinstance(threshold, tuple):
        low, high = threshold
        if low > high:
            raise ValueError(f"Invalid threshold interval: low ({low}) > high ({high})")
        out = out[(s >= low) & (s < high)].copy()
    else:
        out = out[s >= float(threshold)].copy()

    # Keep a small, useful set of columns by default
    if keep_cols is None:
        base = grouping_keys + [subject_col, "group_label", "y", score_col]
        optional = [c for c in ["n_windows", "p_total_std", "lower_q", "upper_q"] if c in out.columns]
        keep_cols = base + optional

    keep_cols = [c for c in keep_cols if c in out.columns]
    return out.loc[:, keep_cols]


def pocket_metrics_from_raw(
    df: pd.DataFrame,
    threshold: Threshold,
    *,
    # preprocessing knobs
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    models: Optional[Union[str, Sequence[str]]] = None,
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[list[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    # metrics knobs
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    One-liner:
      - builds df_hi (threshold pocket)
      - builds df_all (matching full pop)
      - returns 1-row summary (and optionally df_hi)

    Applies drop_subject_ids consistently to BOTH df_hi and df_all.
    """
    df_hi = preprocess_by_threshold(
        df,
        threshold=threshold,
        score_col=score_col,
        split=split,
        models=models,
        variants=variants,
        grouping_keys=grouping_keys,
        enforce_unique=enforce_unique,
        drop_subject_ids=drop_subject_ids,
        subject_col=subject_col,
    )

    df_all = preprocess_by_threshold(
        df,
        threshold=(0.0, 1.0),
        score_col=score_col,
        split=split,
        models=models,
        variants=variants,
        grouping_keys=grouping_keys,
        enforce_unique=enforce_unique,
        drop_subject_ids=drop_subject_ids,
        subject_col=subject_col,
    )

    summary = pocket_metrics_df(
        df_hi=df_hi,
        df_all=df_all,
        threshold=threshold,
        y_col=y_col,
        label_col=label_col,
        score_col=score_col,
        meta_cols=meta_cols,
    )

    return summary, df_hi


def pocket_metrics_by_model(
    df: pd.DataFrame,
    threshold: Threshold,
    *,
    # selection
    model: Optional[Union[str, Sequence[str]]] = None,  # None => all models in df
    # pass-through knobs (match pocket_metrics_from_raw)
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[list[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Run `pocket_metrics_from_raw` separately for each model and return results in a dict.

    Returns
    -------
    dict[model_name, (summary_df, df_hi)]
        - summary_df: 1-row pocket summary for that model (and other filters)
        - df_hi: the threshold-selected pocket rows for that model
    """
    if "model" not in df.columns:
        raise KeyError("df must contain a 'model' column to run per-model pocket metrics.")

    # Resolve which models to run
    if model is None:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    elif isinstance(model, str):
        model_list = [model]
    else:
        model_list = list(model)

    if len(model_list) == 0:
        raise ValueError("No models selected.")

    results: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

    for m in model_list:
        summary, df_hi = pocket_metrics_from_raw(
            df=df,
            threshold=threshold,
            score_col=score_col,
            split=split,
            models=m,                 # <-- key bit: run one model at a time
            variants=variants,
            grouping_keys=grouping_keys,
            enforce_unique=enforce_unique,
            drop_subject_ids=drop_subject_ids,
            subject_col=subject_col,
            y_col=y_col,
            label_col=label_col,
            meta_cols=meta_cols,
        )
        results[str(m)] = (summary, df_hi)

    return results



def threshold_operating_metrics_df(
    df_hi: pd.DataFrame,
    df_all: pd.DataFrame,
    threshold: Threshold,
    *,
    y_col: str = "y",
    score_col: str = "p_mean",
    meta_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a 1-row summary table describing how a threshold behaves as an
    enrichment rule in the full screened population.

    What this function does
    -----------------------
    This function compares:
      - `df_hi`: the patients selected by the threshold
      - `df_all`: the full eligible population before thresholding

    It then summarizes what the threshold is doing from an enrichment and
    screening perspective.

    In other words, this is not only asking:
      "How pure is the selected pocket?"
    but also:
      "What fraction of the full population is selected?"
      "How many positive cases are captured?"
      "How many negative cases are excluded?"
      "How much screening burden does this threshold create?"

    Why this is useful
    ------------------
    The existing pocket summary is helpful for describing the selected subgroup,
    especially its purity (PPV) and enrichment factor. However, for downstream
    trial-planning tasks such as threshold selection, screening burden, and
    sample-size planning, you also need full operating characteristics of the
    threshold.

    This function adds that broader view by reporting:
      - size of the selected subgroup
      - size of the non-selected subgroup
      - positive and negative counts in the full population
      - positive and negative counts among selected and non-selected patients
      - PPV / NPV
      - sensitivity / specificity
      - false discovery rate
      - false negative rate
      - percent selected
      - screen-fail rate
      - number needed to screen (NNS)
      - baseline prevalence and enrichment factor

    Interpretation
    --------------
    Think of this function as turning a model threshold into a screening rule.

    Example:
      If the threshold is 0.70, this function tells you:
        - how many patients would be selected,
        - how many ASD patients are captured,
        - how many TD patients are excluded,
        - how enriched the selected subgroup becomes,
        - and how many patients you would need to screen to obtain one selected
          participant.

    Inputs
    ------
    df_hi:
        Threshold-selected subjects. This is the "enriched pocket", such as
        all patients with score >= 0.70.

    df_all:
        Full eligible population under the same filtering conditions
        (same model, split, variant, exclusions, etc.), before applying the
        threshold.

    threshold:
        Either a single cutoff (e.g., 0.70) or an interval/band
        (e.g., (0.50, 0.70)).

    y_col:
        Column containing the binary outcome. Assumes y=1 is the positive class.

    score_col:
        Name of the score column used to define the threshold.

    meta_cols:
        Metadata columns to carry into the output row, such as model, variant,
        and split.

    Returns
    -------
    pd.DataFrame
        A 1-row dataframe summarizing threshold operating characteristics for
        the specified model / split / variant / threshold.

    Notes
    -----
    This function is designed for threshold evaluation and trial-planning
    support. It complements the pocket summary rather than replacing it:

      - pocket summary:
          describes the selected subgroup itself
      - threshold operating summary:
          describes what the threshold does to the whole screened population
    """
    if meta_cols is None:
        meta_cols = ["model", "variant", "split"]

    if df_hi is None:
        raise ValueError("df_hi is None.")
    if df_all is None or df_all.empty:
        raise ValueError("df_all is empty.")

    if y_col not in df_all.columns:
        raise ValueError(f"df_all missing required column: {y_col}")
    if not df_hi.empty and y_col not in df_hi.columns:
        raise ValueError(f"df_hi missing required column: {y_col}")

    y_all = df_all[y_col].dropna().astype(int)
    y_hi = df_hi[y_col].dropna().astype(int) if not df_hi.empty else pd.Series([], dtype=int)

    # Full population totals:
    # count everyone eligible before applying the threshold,
    # then split that full population into positives (ASD) and negatives (TD)
    n_total = int(len(y_all))
    n_pos_total = int((y_all == 1).sum())
    n_neg_total = int((y_all == 0).sum())

    # Selected:
    # count only the subjects who passed the threshold,
    # then split those selected subjects into positives and negatives
    n_selected = int(len(y_hi))
    n_pos_selected = int((y_hi == 1).sum())   # true positives: 
    n_neg_selected = int((y_hi == 0).sum())   # false positives: 

    # Not selected:
    # derive the excluded group by subtracting selected counts from full-population counts
    n_not_selected = n_total - n_selected
    n_pos_not_selected = n_pos_total - n_pos_selected   # false negatives: ASD patients missed by threshold
    n_neg_not_selected = n_neg_total - n_neg_selected   # true negatives: TD patients correctly excluded

    # Main rates:
    # summarize how the threshold behaves as a screening/enrichment rule
    pct_selected = (n_selected / n_total) if n_total > 0 else float("nan")  # fraction of all eligible subjects selected
    screen_fail_rate = (n_not_selected / n_total) if n_total > 0 else float("nan")  # fraction not selected

    ppv = (n_pos_selected / n_selected) if n_selected > 0 else float("nan")  # among selected, proportion that are truly ASD
    npv = (n_neg_not_selected / n_not_selected) if n_not_selected > 0 else float("nan")  # among not selected, proportion that are truly TD

    sensitivity = (n_pos_selected / n_pos_total) if n_pos_total > 0 else float("nan")  # proportion of all ASD patients captured
    specificity = (n_neg_not_selected / n_neg_total) if n_neg_total > 0 else float("nan")  # proportion of all TD patients correctly excluded

    fdr = (n_neg_selected / n_selected) if n_selected > 0 else float("nan")  # among selected, proportion that are actually TD
    fnr = (n_pos_not_selected / n_pos_total) if n_pos_total > 0 else float("nan")  # proportion of ASD patients missed

    baseline_prevalence = (n_pos_total / n_total) if n_total > 0 else float("nan")  # ASD prevalence before enrichment
    enrichment_factor = (
        ppv / baseline_prevalence
        if pd.notna(ppv) and pd.notna(baseline_prevalence) and baseline_prevalence > 0
        else float("nan")
    )  # how much ASD prevalence increases after thresholding relative to baseline

    nns = (1.0 / pct_selected) if pd.notna(pct_selected) and pct_selected > 0 else float("nan")  # how many candidates must be screened to obtain one selected patient

    # Threshold reporting:
    # store the threshold in a consistent [low, high) format for the summary table
    if isinstance(threshold, tuple):
        thr_low, thr_high = float(threshold[0]), float(threshold[1])
    else:
        thr_low, thr_high = float(threshold), 1.0


    # metadata
    meta: Dict[str, Any] = {}
    for c in meta_cols:
        if c in df_all.columns:
            vals = df_all[c].dropna().unique()
            meta[c] = vals[0] if len(vals) == 1 else list(vals)

    row = {
        **meta,
        "score_col": score_col,
        "thr_low": thr_low,
        "thr_high": thr_high,

        "n_total": n_total,
        "n_pos_total": n_pos_total,
        "n_neg_total": n_neg_total,

        "n_selected": n_selected,
        "n_not_selected": n_not_selected,
        "pct_selected": pct_selected,
        "screen_fail_rate": screen_fail_rate,

        "n_pos_selected": n_pos_selected,
        "n_neg_selected": n_neg_selected,

        "n_pos_not_selected": n_pos_not_selected,
        "n_neg_not_selected": n_neg_not_selected,

        "ppv": ppv,
        "npv": npv,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fdr": fdr,
        "fnr": fnr,

        "baseline_prevalence": baseline_prevalence,
        "enrichment_factor": enrichment_factor,
        "nns": nns,
    }

    return pd.DataFrame([row])


def enrichment_metrics_by_model(
    df: pd.DataFrame,
    threshold: Threshold,
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Wrapper function that runs the full enrichment-threshold analysis separately
    for each model and returns both pocket-level and population-level summaries.

    What this function does
    -----------------------
    For each requested model, this function:

      1. builds the threshold-selected subgroup (`df_hi`)
      2. builds the full eligible population (`df_all`)
      3. computes the current pocket summary
      4. computes the broader threshold operating summary

    This keeps the workflow in one place and avoids manually rebuilding the
    selected and full datasets for each model.

    Why this is useful
    ------------------
    In the enrichment workflow, there are really two different questions:

    1. What does the selected subgroup look like?
       - How many patients were selected?
       - How many positives and negatives are in the selected pocket?
       - What is the PPV / enrichment factor?

    2. What does this threshold do to the full screened population?
       - What fraction of patients are selected?
       - How many positives are captured?
       - How many negatives are excluded?
       - What is the sensitivity / specificity / NPV?
       - How much screening burden does this threshold create?

    Your earlier `pocket_metrics_by_model(...)` wrapper answers mainly the first
    question. This wrapper is meant to answer both at the same time.

    In practice, this means you can evaluate a threshold such as 0.70 for each
    model and directly obtain:
      - a pocket summary for enrichment interpretation
      - an operating summary for threshold selection and later sample-size work
      - the selected subjects themselves
      - the full eligible population used as the denominator

    Typical use case
    ----------------
    Use this function when you want to compare candidate enrichment thresholds
    across one or more models and prepare for downstream tasks such as:

      - threshold selection
      - sensitivity / specificity review
      - screening burden calculations
      - power and sample-size planning

    Inputs
    ------
    df:
        Patient-level dataframe containing model scores and labels, such as
        `df_pat`.

    threshold:
        Threshold definition used for enrichment.
        Examples:
          - 0.70        -> select score >= 0.70
          - (0.50, 0.7) -> select a score band

    model:
        Which model(s) to run. If None, all models in `df` are analyzed.

    score_col:
        Score column used for thresholding, such as `p_mean`.

    split, variants:
        Filters used to define the eligible population before thresholding.

    grouping_keys, enforce_unique, drop_subject_ids, subject_col:
        Passed through to preprocessing so that the selected and full datasets
        are defined consistently.

    y_col, label_col:
        Outcome and label columns.

    meta_cols:
        Metadata columns included in the summary outputs.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary keyed by model name. For each model, the returned object
        contains:

        - "pocket_summary":
            1-row summary of the selected subgroup
        - "operating_summary":
            1-row summary of what the threshold does in the full population
        - "df_hi":
            threshold-selected subjects
        - "df_all":
            full eligible population before thresholding

    Interpretation
    --------------
    This wrapper is designed to make the enrichment analysis easier to use and
    less error-prone.

    Instead of manually:
      - filtering selected subjects,
      - rebuilding the full population,
      - calling multiple summary functions,
      - and keeping model-specific outputs aligned,

    you make one call and receive all enrichment outputs in a consistent format.

    Notes
    -----
    Conceptually, this wrapper separates two levels of analysis:

      - Pocket level:
          "What does the enriched subgroup look like?"
      - Population level:
          "What does this threshold do to the overall screened cohort?"

    That distinction becomes especially important when moving from descriptive
    enrichment results into threshold comparison, screening burden analysis,
    and sample-size planning.
    """
    if "model" not in df.columns:
        raise KeyError("df must contain a 'model' column.")

    # Resolve model list
    if model is None:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    elif isinstance(model, str):
        model_list = [model]
    else:
        model_list = list(model)

    if len(model_list) == 0:
        raise ValueError("No models selected.")

    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for m in model_list:
        # Selected pocket + current summary
        pocket_summary, df_hi = pocket_metrics_from_raw(
            df=df,
            threshold=threshold,
            score_col=score_col,
            split=split,
            models=m,
            variants=variants,
            grouping_keys=grouping_keys,
            enforce_unique=enforce_unique,
            drop_subject_ids=drop_subject_ids,
            subject_col=subject_col,
            y_col=y_col,
            label_col=label_col,
            meta_cols=meta_cols,
        )

        # Full eligible population
        df_all = preprocess_by_threshold(
            df=df,
            threshold=(0.0, 1.0),
            score_col=score_col,
            split=split,
            models=m,
            variants=variants,
            grouping_keys=grouping_keys,
            enforce_unique=enforce_unique,
            drop_subject_ids=drop_subject_ids,
            subject_col=subject_col,
        )

        # New operating summary
        operating_summary = threshold_operating_metrics_df(
            df_hi=df_hi,
            df_all=df_all,
            threshold=threshold,
            y_col=y_col,
            score_col=score_col,
            meta_cols=meta_cols,
        )

        results[str(m)] = {
            "pocket_summary": pocket_summary,
            "operating_summary": operating_summary,
            "df_hi": df_hi,
            "df_all": df_all,
        }

    return results


def ppv_precision_sample_size_from_summary(
    summary_df: pd.DataFrame,
    *,
    ppv_col: str = "ppv",
    pct_selected_col: str = "pct_selected",
    confidence: float = 0.95,
    precision: float = 0.05,
    ceil_n: bool = True,
) -> pd.DataFrame:
    """
    Estimate the required selected sample size and implied screened sample size
    for validating PPV at a fixed enrichment threshold.

    What this function does
    -----------------------
    This function takes a 1-row operating summary table and uses:
      - the observed PPV in the selected subgroup
      - the observed percent selected

    to estimate:

      1. how many selected patients are needed so that PPV can be estimated
         with a desired confidence-interval half-width (`precision`)
      2. how many total patients must be screened to obtain that many selected
         patients, given the observed threshold pass rate (`pct_selected`)

    Why this is useful
    ------------------
    Once the threshold is fixed (for example, 0.70), the next trial-planning
    question is not threshold optimization, but study sizing.

    In this setting, PPV is one of the key enrichment outcomes because it tells
    you how pure the selected subgroup is. This function turns that expected PPV
    into a planning calculation for the next study.

    Interpretation
    --------------
    The output is a planning table. For the chosen threshold, it tells you:

      - required_selected_n:
          how many threshold-selected participants you need
      - implied_screened_n:
          how many total candidates you would need to screen to obtain them

    Parameters
    ----------
    summary_df:
        A 1-row operating summary dataframe containing at least:
          - `ppv`
          - `pct_selected`

    ppv_col:
        Column containing the expected PPV for the selected subgroup.

    pct_selected_col:
        Column containing the fraction of screened patients expected to pass
        the enrichment threshold.

    confidence:
        Confidence level for the PPV confidence interval.
        Default is 0.95 for a 95% CI.

    precision:
        Desired half-width of the confidence interval around PPV.
        Default is 0.05, meaning PPV should be estimated within ±0.05.

    ceil_n:
        If True, round required sample sizes up to the next whole number.

    Returns
    -------
    pd.DataFrame
        A copy of the input 1-row summary with added planning columns:
          - confidence
          - precision
          - z_value
          - required_selected_n
          - implied_screened_n
    """
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df must be a non-empty dataframe.")
    if len(summary_df) != 1:
        raise ValueError("summary_df must contain exactly one row.")
    if ppv_col not in summary_df.columns:
        raise KeyError(f"summary_df missing required column: {ppv_col}")
    if pct_selected_col not in summary_df.columns:
        raise KeyError(f"summary_df missing required column: {pct_selected_col}")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1.")
    if precision <= 0:
        raise ValueError("precision must be > 0.")

    out = summary_df.copy()

    p_raw = float(out.iloc[0][ppv_col])
    pct_selected = float(out.iloc[0][pct_selected_col])

    if not (0.0 <= p_raw <= 1.0):
        raise ValueError(f"{ppv_col} must be in [0, 1]. Got {p_raw}")

    if not (0.0 <= pct_selected <= 1.0):
        raise ValueError(f"{pct_selected_col} must be in [0, 1]. Got {pct_selected}")

    if pct_selected == 0.0:
        raise ValueError(
            f"{pct_selected_col} is 0.0, so no patients are selected and screened sample size cannot be estimated."
        )

    # clip PPV slightly away from the boundaries for planning purposes
    eps = 1e-6
    p = min(max(p_raw, eps), 1.0 - eps)

    alpha = 1.0 - confidence
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    required_selected_n = (z_value ** 2) * p * (1.0 - p) / (precision ** 2)
    implied_screened_n = required_selected_n / pct_selected

    if ceil_n:
        required_selected_n = math.ceil(required_selected_n)
        implied_screened_n = math.ceil(implied_screened_n)

    out["confidence"] = confidence
    out["precision"] = precision
    out["z_value"] = z_value
    out["required_selected_n"] = required_selected_n
    out["implied_screened_n"] = implied_screened_n

    return out


def _binary_power_binomial(
    *,
    n: int,
    p_alt: float,
    p_null: float,
    alpha: float,
    alternative: Literal["larger", "smaller", "two-sided"],
) -> float:
    """
    Binomial power for a binary enrichment endpoint.

    This computes the probability of rejecting H0 under the assumed true
    selected positive rate p_alt.

    The rejection region is defined using an exact binomial test under p_null.
    Power is then computed by summing the binomial probabilities of that
    rejection region under p_alt.
    """
    if alternative == "larger":
        reject_ks = [
            k
            for k in range(n + 1)
            if binomtest(k, n, p=p_null, alternative="greater").pvalue <= alpha
        ]

    elif alternative == "smaller":
        reject_ks = [
            k
            for k in range(n + 1)
            if binomtest(k, n, p=p_null, alternative="less").pvalue <= alpha
        ]

    else:
        reject_ks = [
            k
            for k in range(n + 1)
            if binomtest(k, n, p=p_null, alternative="two-sided").pvalue <= alpha
        ]

    if len(reject_ks) == 0:
        return 0.0

    power = sum(binom.pmf(k, n, p_alt) for k in reject_ks)

    return float(np.clip(power, 0.0, 1.0))


def _binary_power_normal(
    *,
    n: int,
    p_alt: float,
    p_null: float,
    alpha: float,
    alternative: Literal["larger", "smaller", "two-sided"],
) -> float:
    """
    Normal-approximation power for a binary enrichment endpoint.

    This approximates the selected positive rate as normally distributed.

    This is less exact than the binomial method, but can be useful as a quick
    approximation or comparison.
    """
    nd = NormalDist()

    # Standard error under the null defines the rejection boundary.
    se_null = math.sqrt(p_null * (1.0 - p_null) / n)

    # Standard error under the alternative defines variability when p=p_alt.
    se_alt = math.sqrt(p_alt * (1.0 - p_alt) / n)

    if se_null == 0 or se_alt == 0:
        return float("nan")

    if alternative == "larger":
        z_alpha = nd.inv_cdf(1.0 - alpha)
        critical_rate = p_null + z_alpha * se_null

        power = 1.0 - nd.cdf((critical_rate - p_alt) / se_alt)

    elif alternative == "smaller":
        z_alpha = nd.inv_cdf(1.0 - alpha)
        critical_rate = p_null - z_alpha * se_null

        power = nd.cdf((critical_rate - p_alt) / se_alt)

    else:
        z_alpha = nd.inv_cdf(1.0 - alpha / 2.0)

        lower_critical = p_null - z_alpha * se_null
        upper_critical = p_null + z_alpha * se_null

        power_lower = nd.cdf((lower_critical - p_alt) / se_alt)
        power_upper = 1.0 - nd.cdf((upper_critical - p_alt) / se_alt)

        power = power_lower + power_upper

    return float(np.clip(power, 0.0, 1.0))


def binary_enrichment_power(
    *,
    n: int,
    p_alt: float,
    p_null: float,
    alpha: float = 0.05,
    alternative: Literal["larger", "smaller", "two-sided"] = "larger",
    power_endpoint: Literal["binary"] = "binary",
    power_method: Literal["binomial", "normal"] = "binomial",
) -> float:
    """
    Compute power for a binary enrichment endpoint.

    This function supports the current enrichment use case where the outcome is
    binary.

    Examples
    --------
    Diagnostic enrichment:
        y = 1 -> disease case
        y = 0 -> control

    Prognostic enrichment:
        y = 1 -> responder
        y = 0 -> non-responder

    The default hypothesis for enrichment is:

        H0: selected positive rate = benchmark positive rate
        H1: selected positive rate > benchmark positive rate

    For prognostic enrichment, this becomes:

        H0: selected response rate = baseline response rate
        H1: selected response rate > baseline response rate

    Parameters
    ----------
    n:
        Number of selected participants used for the power calculation.

    p_alt:
        Assumed true positive rate in the selected subgroup.

        For prognostic enrichment, this is the assumed selected response rate.

    p_null:
        Benchmark/null positive rate.

        For prognostic enrichment, this is the benchmark or baseline response rate.

    alpha:
        Type I error rate. Default is 0.05.

    alternative:
        Direction of the test.

        "larger":
            Tests whether p_alt > p_null. This is the usual enrichment case.

        "smaller":
            Tests whether p_alt < p_null.

        "two-sided":
            Tests whether p_alt != p_null.

    power_endpoint:
        Endpoint type. Currently only "binary" is supported.

    power_method:
        Method used for the binary power calculation.

        "binomial":
            Uses the binomial distribution for the number of positive outcomes.
            This is the recommended default for binary enrichment endpoints.

        "normal":
            Uses a normal approximation to the selected positive rate.

    Returns
    -------
    float
        Estimated statistical power, between 0 and 1.
    """
    if power_endpoint != "binary":
        raise ValueError("Only power_endpoint='binary' is currently supported.")

    if not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n <= 0:
        raise ValueError("n must be > 0.")

    if not (0.0 <= p_alt <= 1.0):
        raise ValueError(f"p_alt must be in (0.0, 1.0). Got {p_alt}.")
    if not (0.0 <= p_null <= 1.0):
        raise ValueError(f"p_null must be in (0.0, 1.0). Got {p_null}.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in (0.0, 1.0). Got {alpha}.")

    if alternative not in {"larger", "smaller", "two-sided"}:
        raise ValueError("alternative must be 'larger', 'smaller', or 'two-sided'.")

    if power_method not in {"binomial", "normal"}:
        raise ValueError("power_method must be 'binomial' or 'normal'.")

    if power_method == "binomial":
        return _binary_power_binomial(
            n=n,
            p_alt=p_alt,
            p_null=p_null,
            alpha=alpha,
            alternative=alternative,
        )

    return _binary_power_normal(
        n=n,
        p_alt=p_alt,
        p_null=p_null,
        alpha=alpha,
        alternative=alternative,
    )


def enrichment_power_from_summary(
    summary_df: pd.DataFrame,
    *,
    ppv_col: str = "ppv",
    baseline_col: str = "baseline_prevalence",
    n_selected_col: str = "n_selected",
    pct_selected_col: str = "pct_selected",
    assumed_enriched_rate: Optional[float] = None,
    benchmark_rate: Optional[float] = None,
    selected_n: Optional[int] = None,
    alpha: float = 0.05,
    alternative: Literal["larger", "smaller", "two-sided"] = "larger",
    power_endpoint: Literal["binary"] = "binary",
    power_method: Literal["binomial", "normal"] = "binomial",
) -> pd.DataFrame:
    """
    Add enrichment power calculations to a 1-row operating or planning summary.

    This function is generic and can be used for both diagnostic and prognostic
    enrichment.

    What this function tests
    ------------------------
    The default enrichment power question is:

        Is the selected subgroup positive rate higher than the baseline
        positive rate in the full eligible population?

    Diagnostic interpretation:
        Is the selected disease-case rate higher than the baseline disease
        prevalence?

    Prognostic interpretation:
        Is the selected response rate higher than the baseline response rate?

    Default behavior
    ----------------
    If no overrides are provided:

        assumed_enriched_rate = summary_df[ppv_col]
        benchmark_rate        = summary_df[baseline_col]
        selected_n            = summary_df[n_selected_col]

    This means the default power calculation uses the observed selected
    subgroup rate, the observed baseline rate, and the observed selected N.

    Endpoint and method
    -------------------
    Currently, only binary enrichment endpoints are supported.

    For binary endpoints, the method can be:

        "binomial":
            Uses the binomial distribution for the number of positive outcomes
            among selected participants. This is the recommended default.

        "normal":
            Uses a normal approximation to the selected positive rate.

    Parameters
    ----------
    summary_df:
        A 1-row summary dataframe, usually the operating summary or planning
        summary from the enrichment pipeline.

    ppv_col:
        Column containing the selected subgroup positive rate.

    baseline_col:
        Column containing the full-population baseline positive rate.

    n_selected_col:
        Column containing the number of selected participants.

    pct_selected_col:
        Column containing the fraction of screened candidates selected by the
        threshold. Used to compute implied screened N.

    assumed_enriched_rate:
        Optional assumed selected subgroup positive rate for planning.
        If None, uses observed ppv_col.

    benchmark_rate:
        Optional benchmark/null positive rate.
        If None, uses baseline_col.

    selected_n:
        Optional selected sample size for the power calculation.
        If None, uses observed n_selected_col.

    alpha:
        Type I error rate. Default is 0.05.

    alternative:
        Test direction. Default is "larger", which is the usual enrichment
        question.

    power_endpoint:
        Endpoint type for power analysis. Currently only "binary" is supported.

    power_method:
        Method used for binary endpoint power.

        "binomial":
            Uses the binomial distribution.

        "normal":
            Uses a normal approximation.

    Returns
    -------
    pd.DataFrame
        Copy of summary_df with added power-planning columns:
          - power_alpha
          - power_alternative
          - power_endpoint
          - power_method
          - power_benchmark_rate
          - power_assumed_enriched_rate
          - power_selected_n
          - power
          - power_absolute_lift
          - power_relative_enrichment
          - power_implied_screened_n
    """
    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df must be a non-empty dataframe.")
    if len(summary_df) != 1:
        raise ValueError("summary_df must contain exactly one row.")

    required = [ppv_col, baseline_col, n_selected_col, pct_selected_col]
    missing = [c for c in required if c not in summary_df.columns]
    if missing:
        raise KeyError(f"summary_df missing required columns: {missing}")

    out = summary_df.copy()

    observed_ppv = float(out.iloc[0][ppv_col])
    observed_baseline = float(out.iloc[0][baseline_col])
    observed_n_selected = int(out.iloc[0][n_selected_col])
    pct_selected = float(out.iloc[0][pct_selected_col])

    p_alt = observed_ppv if assumed_enriched_rate is None else float(assumed_enriched_rate)
    p_null = observed_baseline if benchmark_rate is None else float(benchmark_rate)
    n_power = observed_n_selected if selected_n is None else int(selected_n)

    power = binary_enrichment_power(
        n=n_power,
        p_alt=p_alt,
        p_null=p_null,
        alpha=alpha,
        alternative=alternative,
        power_endpoint=power_endpoint,
        power_method=power_method,
    )

    implied_screened_n = (
        math.ceil(n_power / pct_selected)
        if pct_selected > 0 and pd.notna(pct_selected)
        else float("nan")
    )

    out["power_alpha"] = alpha
    out["power_alternative"] = alternative
    out["power_endpoint"] = power_endpoint
    out["power_method"] = power_method
    out["power_benchmark_rate"] = p_null
    out["power_assumed_enriched_rate"] = p_alt
    out["power_selected_n"] = n_power
    out["power"] = power
    out["power_absolute_lift"] = p_alt - p_null
    out["power_relative_enrichment"] = p_alt / p_null if p_null > 0 else float("nan")
    out["power_implied_screened_n"] = implied_screened_n

    return out


def enrichment_pipeline_by_model(
    df: pd.DataFrame,
    threshold: Threshold,
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
    confidence: float = 0.95,
    precision: float = 0.05,
    ceil_n: bool = True,
    compute_power: bool = True,
    power_alpha: float = 0.05,
    power_alternative: Literal["larger", "smaller", "two-sided"] = "larger",
    power_endpoint: Literal["binary"] = "binary",
    power_method: Literal["binomial", "normal"] = "binomial",
    power_assumed_enriched_rate: Optional[float] = None,
    power_benchmark_rate: Optional[float] = None,
    power_selected_n: Optional[int] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run the full enrichment workflow separately for each model and return all
    threshold, operating, and planning outputs in one place.

    What this function does
    -----------------------
    For each requested model, this wrapper:

      1. builds the threshold-selected subgroup (`df_hi`)
      2. builds the full eligible population (`df_all`)
      3. computes the pocket summary
      4. computes the threshold operating summary
      5. computes precision-based sample-size planning
      6. optionally computes power-based planning

    This is the top-level enrichment wrapper. It replaces the need to call:
      - `pocket_metrics_by_model(...)`
      - `enrichment_metrics_by_model(...)`
      - `ppv_precision_sample_size_from_summary(...)`
      - `enrichment_power_from_summary(...)`
    separately.

    Why this is useful
    ------------------
    The enrichment workflow has three layers:

      - Pocket summary:
          describes the selected subgroup itself

      - Operating summary:
          describes what the threshold does to the full screened population

      - Planning summary:
          estimates how many selected and screened patients are needed in the
          next study and, when requested, estimates power to detect enrichment
          relative to a benchmark rate.

    This wrapper keeps those outputs aligned and makes the workflow easier to
    use and less error-prone.

    Power planning
    --------------
    If compute_power=True, the planning summary is augmented with a power
    calculation.

    For the current implementation, power_endpoint must be "binary".

    Binary endpoint examples:
      - diagnostic enrichment: disease vs control
      - prognostic enrichment: responder vs non-responder

    For binary endpoints, power_method controls the statistical calculation:

      - "binomial":
          Uses the binomial distribution for the number of positive outcomes.
          This is the recommended default for binary enrichment endpoints.

      - "normal":
          Uses a normal approximation to the selected positive rate.

    By default, the power calculation uses:

      - observed PPV as the assumed enriched subgroup positive rate
      - observed baseline prevalence as the benchmark/null positive rate
      - observed selected N as the selected sample size

    These defaults can be overridden with:

      - power_assumed_enriched_rate
      - power_benchmark_rate
      - power_selected_n

    Parameters
    ----------
    df:
        Patient-level prediction dataframe.

    threshold:
        Threshold used to define the enriched subgroup.

        If a float is provided, subjects with score_col >= threshold are selected.

        If a tuple is provided, it is interpreted as an interval:
            low <= score_col < high

    model:
        Model name or sequence of model names to analyze.
        If None, all models present in df are analyzed.

    score_col:
        Patient-level prediction column used for thresholding.

    split:
        Split to analyze, such as "test" or "external".
        If None, no split filtering is applied.

    variants:
        Prediction or calibration variant to analyze, such as "beta".
        If None, all variants are included.

    grouping_keys:
        Columns defining an evaluation context for uniqueness checks.

    enforce_unique:
        If True, enforce one row per subject per evaluation context before
        applying the threshold.

    drop_subject_ids:
        Optional subject IDs to exclude from both df_hi and df_all.

    subject_col:
        Subject identifier column.

    y_col:
        Binary outcome column. y=1 is treated as the positive class.

    label_col:
        Optional human-readable label column corresponding to y_col.

    meta_cols:
        Metadata columns to retain in summary outputs.

    confidence:
        Confidence level used for precision-based planning.

    precision:
        Desired half-width for estimating PPV / selected positive rate.

    ceil_n:
        If True, round required sample sizes up to the nearest integer.

    compute_power:
        If True, add power-based planning columns to planning_summary.

    power_alpha:
        Type I error rate for the power calculation.

    power_alternative:
        Direction of the power test.

        "larger" is the usual enrichment setting, testing whether the selected
        subgroup positive rate is higher than the benchmark positive rate.

    power_endpoint:
        Endpoint type for power analysis. Currently only "binary" is supported.

    power_method:
        Method used for binary endpoint power.

        "binomial" uses the binomial distribution.
        "normal" uses a normal approximation.

    power_assumed_enriched_rate:
        Optional assumed selected subgroup positive rate for power planning.
        If None, uses the observed PPV.

    power_benchmark_rate:
        Optional benchmark/null positive rate for power planning.
        If None, uses the observed baseline prevalence.

    power_selected_n:
        Optional selected sample size for power planning.
        If None, uses the observed n_selected.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary keyed by model name. For each model, the returned object
        contains:

        - "pocket_summary":
            1-row summary of the selected subgroup

        - "operating_summary":
            1-row summary of threshold operating characteristics

        - "planning_summary":
            1-row planning table containing precision-based planning and,
            when compute_power=True, power-based planning

        - "df_hi":
            threshold-selected subjects

        - "df_all":
            full eligible population before thresholding
    """
    if "model" not in df.columns:
        raise KeyError("df must contain a 'model' column.")

    if model is None:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    elif isinstance(model, str):
        model_list = [model]
    else:
        model_list = list(model)

    if len(model_list) == 0:
        raise ValueError("No models selected.")

    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for m in model_list:
        #-------------------------------------------------------
        # Step 1: Build the threshold-selected subgroup and its
        # pocket summary. This describes the enriched subgroup
        # after applying the threshold.
        #
        # Examples:
        #   diagnostic enrichment:
        #       selected subgroup enriched for disease cases
        #
        #   prognostic enrichment:
        #       selected subgroup enriched for responders
        #-------------------------------------------------------
        pocket_summary, df_hi = pocket_metrics_from_raw(
            df=df,
            threshold=threshold,
            score_col=score_col,
            split=split,
            models=m,
            variants=variants,
            grouping_keys=grouping_keys,
            enforce_unique=enforce_unique,
            drop_subject_ids=drop_subject_ids,
            subject_col=subject_col,
            y_col=y_col,
            label_col=label_col,
            meta_cols=meta_cols,
        )

        #-------------------------------------------------------
        # Step 2: Build the full eligible population before
        # thresholding. This is the denominator population used
        # to understand what the threshold is doing overall.
        #-------------------------------------------------------
        df_all = preprocess_by_threshold(
            df=df,
            threshold=(0.0, 1.0),
            score_col=score_col,
            split=split,
            models=m,
            variants=variants,
            grouping_keys=grouping_keys,
            enforce_unique=enforce_unique,
            drop_subject_ids=drop_subject_ids,
            subject_col=subject_col,
        )

        #-------------------------------------------------------
        # Step 3: Compute threshold operating characteristics.
        #
        # This uses the selected subgroup (df_hi) and the full
        # eligible population (df_all) to summarize how the
        # threshold behaves as a screening / enrichment rule.
        #
        # Outputs include:
        #   - percent selected
        #   - screen-fail rate
        #   - PPV / selected positive rate
        #   - NPV
        #   - sensitivity
        #   - specificity
        #   - enrichment factor
        #   - number needed to screen
        #-------------------------------------------------------
        operating_summary = threshold_operating_metrics_df(
            df_hi=df_hi,
            df_all=df_all,
            threshold=threshold,
            y_col=y_col,
            score_col=score_col,
            meta_cols=meta_cols,
        )

        #-------------------------------------------------------
        # Step 4: Precision-based planning.
        #
        # This estimates how many selected subjects are needed
        # to estimate PPV / selected positive rate with the
        # desired confidence interval precision, and how many
        # total candidates would need to be screened to obtain
        # that selected N.
        #-------------------------------------------------------
        planning_summary = ppv_precision_sample_size_from_summary(
            operating_summary,
            confidence=confidence,
            precision=precision,
            ceil_n=ceil_n,
        )

        #-------------------------------------------------------
        # Step 5: Optional power-based planning.
        #
        # compute_power is the on/off switch.
        #
        # If compute_power=True, the remaining power_* parameters
        # are passed into enrichment_power_from_summary:
        #
        #   power_alpha                 -> alpha
        #   power_alternative           -> alternative
        #   power_endpoint              -> power_endpoint
        #   power_method                -> power_method
        #   power_assumed_enriched_rate -> assumed_enriched_rate
        #   power_benchmark_rate        -> benchmark_rate
        #   power_selected_n            -> selected_n
        #
        # The power helper expects generic enrichment column names:
        #
        #   ppv
        #   baseline_prevalence
        #   n_selected
        #   pct_selected
        #
        # Therefore, power is added here, before any downstream
        # diagnostic- or prognostic-specific renaming.
        #-------------------------------------------------------
        if compute_power:
            planning_summary = enrichment_power_from_summary(
                planning_summary,
                ppv_col="ppv",
                baseline_col="baseline_prevalence",
                n_selected_col="n_selected",
                pct_selected_col="pct_selected",
                assumed_enriched_rate=power_assumed_enriched_rate,
                benchmark_rate=power_benchmark_rate,
                selected_n=power_selected_n,
                alpha=power_alpha,
                alternative=power_alternative,
                power_endpoint=power_endpoint,
                power_method=power_method,
            )

        #-------------------------------------------------------
        # Step 6: Store all outputs for this model in one place.
        #-------------------------------------------------------
        results[str(m)] = {
            "pocket_summary": pocket_summary,
            "operating_summary": operating_summary,
            "planning_summary": planning_summary,
            "df_hi": df_hi,
            "df_all": df_all,
        }

    return results


def diagnostic_enrichment_pipeline_by_model(
    df: pd.DataFrame,
    threshold: Threshold,
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
    confidence: float = 0.95,
    precision: float = 0.05,
    ceil_n: bool = True,
    compute_power: bool = True,
    power_alpha: float = 0.05,
    power_alternative: Literal["larger", "smaller", "two-sided"] = "larger",
    power_endpoint: Literal["binary"] = "binary",
    power_method: Literal["binomial", "normal"] = "binomial",
    power_assumed_enriched_rate: Optional[float] = None,
    power_benchmark_rate: Optional[float] = None,
    power_selected_n: Optional[int] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run diagnostic enrichment analysis separately by model.

    This function is a diagnostic-specific wrapper around the generic
    `enrichment_pipeline_by_model(...)`.

    Interpretation
    --------------
    This wrapper is intended for diagnostic enrichment, where the binary outcome
    is interpreted as:

        y = 1  -> disease case / diagnostic positive class
        y = 0  -> control / diagnostic negative class

    The score column, usually `p_mean`, is interpreted as a patient-level
    predicted probability of belonging to the diagnostic positive class.

    Patients with score >= threshold are selected into the diagnostic-enriched
    subgroup.

    What this function does
    -----------------------
    For each requested model, this wrapper calls the generic enrichment engine to:

      1. build the threshold-selected subgroup
      2. build the full eligible population
      3. compute the pocket summary
      4. compute the threshold operating summary
      5. compute precision-based sample-size planning
      6. optionally compute binary-endpoint power planning

    Notes
    -----
    Unlike `prognostic_enrichment_pipeline_by_model(...)`, this diagnostic
    wrapper does not rename columns into response-specific terminology.

    Generic columns such as:

        ppv
        baseline_prevalence
        enrichment_factor
        sensitivity
        specificity

    are appropriate for diagnostic enrichment, where the positive class is the
    diagnostic target.

    Parameters
    ----------
    df:
        Patient-level prediction dataframe.

    threshold:
        Threshold used to define the diagnostic-enriched subgroup.

    model:
        Model name or sequence of model names to analyze. If None, all models
        present in df are analyzed.

    score_col:
        Patient-level predicted probability column used for thresholding.

    split:
        Split to analyze, such as "test" or "external".

    variants:
        Prediction or calibration variant to analyze, such as "beta".

    grouping_keys:
        Columns defining an evaluation context for uniqueness checks.

    enforce_unique:
        If True, enforce one row per subject per evaluation context before
        thresholding.

    drop_subject_ids:
        Optional subject IDs to exclude from both selected and full populations.

    subject_col:
        Subject identifier column.

    y_col:
        Binary diagnostic outcome column. y=1 is interpreted as the diagnostic
        positive class.

    label_col:
        Optional human-readable label column corresponding to y_col.

    meta_cols:
        Metadata columns to retain in summary outputs.

    confidence:
        Confidence level for precision-based planning.

    precision:
        Desired half-width for estimating PPV / selected positive rate.

    ceil_n:
        If True, round required sample sizes up to the nearest integer.

    compute_power:
        If True, add power-based planning columns to planning_summary.

    power_alpha:
        Type I error rate for power calculation.

    power_alternative:
        Direction of the power test. For enrichment, "larger" is usually used.

    power_endpoint:
        Endpoint type for power analysis. Currently only "binary" is supported.

    power_method:
        Method used for binary endpoint power.

        "binomial":
            Uses the binomial distribution for the number of positive outcomes.

        "normal":
            Uses a normal approximation to the selected positive rate.

    power_assumed_enriched_rate:
        Optional assumed selected positive rate for power planning.
        If None, uses observed PPV.

    power_benchmark_rate:
        Optional benchmark/null positive rate for power planning.
        If None, uses observed baseline prevalence.

    power_selected_n:
        Optional selected sample size for power planning.
        If None, uses observed n_selected.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary keyed by model name.

        Each model block contains:

          - pocket_summary
          - operating_summary
          - planning_summary
          - df_hi
          - df_all
    """

    # ------------------------------------------------------------------
    # Diagnostic enrichment is a thin wrapper around the generic
    # enrichment engine.
    #
    # The generic engine already computes:
    #   - selected subgroup
    #   - full denominator population
    #   - pocket summary
    #   - operating summary
    #   - precision planning
    #   - optional power planning
    #
    # This wrapper exists so notebook code can call a diagnostic-specific
    # function name, while keeping all core calculations centralized in
    # enrichment_pipeline_by_model(...).
    # ------------------------------------------------------------------
    return enrichment_pipeline_by_model(
        df=df,
        threshold=threshold,
        model=model,
        score_col=score_col,
        split=split,
        variants=variants,
        grouping_keys=grouping_keys,
        enforce_unique=enforce_unique,
        drop_subject_ids=drop_subject_ids,
        subject_col=subject_col,
        y_col=y_col,
        label_col=label_col,
        meta_cols=meta_cols,
        confidence=confidence,
        precision=precision,
        ceil_n=ceil_n,
        compute_power=compute_power,
        power_alpha=power_alpha,
        power_alternative=power_alternative,
        power_endpoint=power_endpoint,
        power_method=power_method,
        power_assumed_enriched_rate=power_assumed_enriched_rate,
        power_benchmark_rate=power_benchmark_rate,
        power_selected_n=power_selected_n,
    )


# ---------------------------------------------------------------------
# Column renaming for prognostic enrichment for treatment response
# ---------------------------------------------------------------------
# The generic enrichment pipeline is written for binary enrichment where
# y=1 is the positive class and y=0 is the negative class.
#
# In prognostic enrichment for treatment response, we interpret:
#
#   y=1 -> responder
#   y=0 -> non-responder
#
# Therefore, generic columns such as ppv, fdr, sensitivity, and
# baseline_prevalence should be renamed so the output tables directly
# communicate treatment-response meaning.
# ---------------------------------------------------------------------
RESPONSE_ENRICHMENT_RENAME_MAP: Dict[str, str] = {
    # Full-population counts
    "n_pos_total": "n_responders_total",
    "n_neg_total": "n_nonresponders_total",

    # Threshold-selected subgroup counts
    "n_pos_selected": "n_responders_selected",
    "n_neg_selected": "n_nonresponders_selected",

    # Threshold-not-selected subgroup counts
    "n_pos_not_selected": "n_responders_not_selected",
    "n_neg_not_selected": "n_nonresponders_not_selected",

    # Selected / non-selected subgroup rates
    "ppv": "selected_response_rate",
    "fdr": "selected_nonresponse_rate",
    "npv": "nonselected_nonresponse_rate",

    # Operating characteristics
    "sensitivity": "responder_capture_rate",
    "specificity": "nonresponder_exclusion_rate",
    "fnr": "responder_miss_rate",

    # Enrichment / screening burden
    "baseline_prevalence": "baseline_response_rate",
    "enrichment_factor": "response_enrichment_factor",
    "nns": "number_needed_to_screen",
}


# ---------------------------------------------------------------------
# Planning-specific renaming
# ---------------------------------------------------------------------
# The generic planning table estimates how many selected participants are
# needed to estimate PPV with the requested confidence and precision, and
# how many total candidates must be screened to obtain that selected N.
#
# For prognostic enrichment, PPV is interpreted as the selected response
# rate. Therefore, planning names should also be response-specific.
# ---------------------------------------------------------------------
RESPONSE_PLANNING_RENAME_MAP = {
    # Precision planning
    "precision": "response_rate_precision",
    "required_selected_n": "required_selected_response_enriched_n",
    "implied_screened_n": "implied_screened_candidates_n",

    # Power planning
    "power_benchmark_rate": "power_benchmark_response_rate",
    "power_assumed_enriched_rate": "power_assumed_selected_response_rate",
    "power_selected_n": "power_selected_response_enriched_n",
    "power_absolute_lift": "power_absolute_response_rate_lift",
    "power_relative_enrichment": "power_relative_response_enrichment",
    "power_implied_screened_n": "power_implied_screened_candidates_n",
}


# ---------------------------------------------------------------------
# Combined map
# ---------------------------------------------------------------------
# Use this when renaming planning_summary, because planning_summary is a
# copy of operating_summary with additional planning columns added.
# ---------------------------------------------------------------------
RESPONSE_PLANNING_FULL_RENAME_MAP: Dict[str, str] = {
    **RESPONSE_ENRICHMENT_RENAME_MAP,
    **RESPONSE_PLANNING_RENAME_MAP,
}


def rename_enrichment_columns_for_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename generic binary-enrichment columns into treatment-response terminology.

    For prognostic enrichment:

        y = 1 -> responder
        y = 0 -> non-responder
    """
    return df.rename(columns=RESPONSE_ENRICHMENT_RENAME_MAP).copy()


def rename_planning_columns_for_response(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename generic planning-summary columns into treatment-response terminology.

    The planning summary contains both:
      - generic enrichment columns
      - precision/power planning columns
    """
    return df.rename(columns=RESPONSE_PLANNING_FULL_RENAME_MAP).copy()


def prognostic_enrichment_pipeline_by_model(
    df: pd.DataFrame,
    threshold: Threshold,
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[list[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
    confidence: float = 0.95,
    precision: float = 0.05,
    ceil_n: bool = True,
    compute_power: bool = True,
    power_alpha: float = 0.05,
    power_alternative: Literal["larger", "smaller", "two-sided"] = "larger",
    power_endpoint: Literal["binary"] = "binary",
    power_method: Literal["binomial", "normal"] = "binomial",
    power_assumed_enriched_rate: Optional[float] = None,
    power_benchmark_rate: Optional[float] = None,
    power_selected_n: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run prognostic enrichment analysis for treatment response, separately by model.

    This function is a treatment-response-specific wrapper around the generic
    threshold-based enrichment pipeline.

    The generic enrichment pipeline computes:
      - threshold-selected subgroup
      - full eligible population
      - pocket summary
      - operating summary
      - precision-based planning summary

    This wrapper adds optional power-based planning and then renames the summary
    columns into treatment-response terminology.

    Interpretation
    --------------
    This function assumes the binary outcome is a response outcome:

        y = 1  -> responder
        y = 0  -> non-responder

    The score column, usually `p_mean`, is interpreted as a patient-level
    predicted probability of response:

        score_col = P(response | baseline features)

    Patients with score >= threshold are selected into the response-enriched
    subgroup.

    This is prognostic enrichment, not predictive treatment-effect enrichment.

    It asks:

        "Is this patient likely to respond?"

    It does not ask:

        "Is this patient more likely to benefit from treatment compared with
        control or no treatment?"

    Power planning
    --------------
    If compute_power=True, this function adds a power calculation to each
    model's planning summary.

    By default, power is computed using:

        power_assumed_enriched_rate = observed selected response rate
        power_benchmark_rate        = observed baseline response rate
        power_selected_n            = observed number selected

    The user may override these values.

    For example, the user can set:

        power_assumed_enriched_rate = 0.80

    to ask how much power the study would have if the true selected response
    rate were 80%, even if the observed selected response rate is different.

    Parameters
    ----------
    df:
        Patient-level prediction dataframe.

    threshold:
        Threshold used to define the response-enriched subgroup.

        If a float is provided:
            selected if score_col >= threshold

        If a tuple is provided:
            selected if low <= score_col < high

    model:
        Model name or sequence of model names to analyze. If None, all models
        present in df are analyzed.

    score_col:
        Patient-level prediction column used for thresholding.

    split:
        Split to analyze, such as "test" or "external".

    variants:
        Prediction/calibration variant to analyze, such as "beta".

    grouping_keys:
        Columns defining an evaluation context for uniqueness checks.

    enforce_unique:
        If True, enforce one row per subject per evaluation context.

    drop_subject_ids:
        Optional subject IDs to exclude.

    subject_col:
        Subject identifier column.

    y_col:
        Binary response outcome column. y=1 is interpreted as responder.

    label_col:
        Optional human-readable label column.

    meta_cols:
        Metadata columns to retain in summaries.

    confidence:
        Confidence level for precision-based planning.

    precision:
        Desired half-width for selected response-rate precision.

    ceil_n:
        Whether to round required sample sizes up.

    compute_power:
        If True, add power-based planning columns to planning_summary.

    power_alpha:
        Type I error rate for the power calculation.

    power_alternative:
        Test direction for power calculation. Usually "larger" for enrichment.

    power_endpoint:
        Endpoint type for power analysis. Currently only "binary" is supported.

    power_method:
        Method used for binary endpoint power.

        "binomial" uses the binomial distribution.
        "normal" uses a normal approximation.

    power_assumed_enriched_rate:
        Optional assumed selected response rate for power planning.
        If None, uses observed selected response rate.

    power_benchmark_rate:
        Optional benchmark response rate for power planning.
        If None, uses observed baseline response rate.

    power_selected_n:
        Optional selected sample size for power planning.
        If None, uses observed n_selected.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary keyed by model name.

        Each model block contains:
          - pocket_summary
          - operating_summary
          - planning_summary
          - df_hi
          - df_all
    """

    # ------------------------------------------------------------------
    # Step 1. Run the generic threshold-based enrichment pipeline
    # ------------------------------------------------------------------
    # This function already performs the core enrichment work:
    #
    #   - filters the analysis population by split / model / variant
    #   - applies the probability threshold
    #   - creates df_hi, the selected subgroup
    #   - creates df_all, the full eligible denominator population
    #   - computes pocket_summary
    #   - computes operating_summary
    #   - computes precision-based planning_summary
    #
    # Because this wrapper is intended to live inside post_analysis.py,
    # call enrichment_pipeline_by_model(...) directly.
    # Do not call post.enrichment_pipeline_by_model(...).
    # ------------------------------------------------------------------
    out = enrichment_pipeline_by_model(
        df=df,
        threshold=threshold,
        model=model,
        score_col=score_col,
        split=split,
        variants=variants,
        grouping_keys=grouping_keys,
        enforce_unique=enforce_unique,
        drop_subject_ids=drop_subject_ids,
        subject_col=subject_col,
        y_col=y_col,
        label_col=label_col,
        meta_cols=meta_cols,
        confidence=confidence,
        precision=precision,
        ceil_n=ceil_n,
        compute_power=compute_power,
        power_alpha=power_alpha,
        power_alternative=power_alternative,
        power_endpoint=power_endpoint,
        power_method=power_method,
        power_assumed_enriched_rate=power_assumed_enriched_rate,
        power_benchmark_rate=power_benchmark_rate,
        power_selected_n=power_selected_n,
    )


    # ------------------------------------------------------------------
    # Step 2. Rename summaries into prognostic treatment-response language
    # ------------------------------------------------------------------
    # After the generic calculations are complete, rename the summary
    # columns so the notebook outputs are directly interpretable as
    # prognostic enrichment for treatment response.
    #
    # The subject-level dataframes df_hi and df_all are not renamed here.
    # ------------------------------------------------------------------
    for model_name, block in out.items():
        block["pocket_summary"] = rename_enrichment_columns_for_response(
            block["pocket_summary"]
        )

        block["operating_summary"] = rename_enrichment_columns_for_response(
            block["operating_summary"]
        )

        block["planning_summary"] = rename_planning_columns_for_response(
            block["planning_summary"]
        )

    return out



# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Stratification workflow
# ------------------------------------------------------------------------------------------------------------------------------------------------------

DIAGNOSTIC_STRATIFICATION_RENAME_MAP: Dict[str, str] = {
    # Full-population counts
    "n_pos_total": "n_diagnostic_positive_total",
    "n_neg_total": "n_diagnostic_negative_total",

    # Stratum counts
    "n_pos_stratum": "n_diagnostic_positive_stratum",
    "n_neg_stratum": "n_diagnostic_negative_stratum",

    # Stratum rates
    "baseline_positive_rate": "baseline_diagnostic_positive_rate",
    "stratum_positive_rate": "stratum_diagnostic_positive_rate",
    "stratum_negative_rate": "stratum_diagnostic_negative_rate",
    "stratum_enrichment_factor": "diagnostic_enrichment_factor",
}

def rename_stratification_columns_for_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename generic stratification columns into diagnostic-stratification terminology.

    Generic stratification assumes:

        y = 1 -> positive class
        y = 0 -> negative class

    For diagnostic stratification, this is interpreted as:

        y = 1 -> diagnostic positive / disease case
        y = 0 -> diagnostic negative / control

    Parameters
    ----------
    df:
        Output from stratification_summary_by_model(...).

    Returns
    -------
    pd.DataFrame
        Copy of df with diagnostic-specific column names.
    """
    return df.rename(columns=DIAGNOSTIC_STRATIFICATION_RENAME_MAP).copy()


PROGNOSTIC_STRATIFICATION_RENAME_MAP: Dict[str, str] = {
    # Full-population counts
    "n_pos_total": "n_responders_total",
    "n_neg_total": "n_nonresponders_total",

    # Stratum counts
    "n_pos_stratum": "n_responders_stratum",
    "n_neg_stratum": "n_nonresponders_stratum",

    # Stratum rates
    "baseline_positive_rate": "baseline_response_rate",
    "stratum_positive_rate": "stratum_response_rate",
    "stratum_negative_rate": "stratum_nonresponse_rate",
    "stratum_enrichment_factor": "response_enrichment_factor",
}


def rename_stratification_columns_for_prognostic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename generic stratification columns into prognostic-response terminology.

    Generic stratification assumes:

        y = 1 -> positive class
        y = 0 -> negative class

    For prognostic stratification for treatment response, this is interpreted as:

        y = 1 -> responder
        y = 0 -> non-responder

    Parameters
    ----------
    df:
        Output from stratification_summary_by_model(...).

    Returns
    -------
    pd.DataFrame
        Copy of df with prognostic response-specific column names.
    """
    return df.rename(columns=PROGNOSTIC_STRATIFICATION_RENAME_MAP).copy()


def stratification_summary_by_model(
    df: pd.DataFrame,
    strata: Sequence[Stratum],
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a patient-stratification summary table for one or more models.

    This function divides the screened population into score-based strata,
    such as low / medium / high probability groups, and summarizes the outcome
    composition within each stratum.

    The function is generic. It does not assume diagnostic or prognostic
    interpretation. It only assumes that y=1 is the positive class and y=0 is
    the negative class.

    Examples
    --------
    Diagnostic stratification:
        score_col = predicted probability of disease
        y = 1     = disease case
        y = 0     = control

    Prognostic stratification:
        score_col = predicted probability of response
        y = 1     = responder
        y = 0     = non-responder

    Parameters
    ----------
    df:
        Patient-level prediction dataframe.

    strata:
        Sequence of named score intervals.

        Each stratum should be a tuple:
            (stratum_name, score_low, score_high)

        The interval is interpreted as:
            score_low <= score_col < score_high

        Example:
            [
                ("low", 0.00, 0.30),
                ("medium", 0.30, 0.70),
                ("high", 0.70, 1.00),
            ]

    model:
        Model name or sequence of model names to summarize.
        If None, all models in df are summarized.

    score_col:
        Patient-level score used for stratification.

    split:
        Split to analyze, such as "test" or "external".
        If None, no split filtering is applied.

    variants:
        Prediction/calibration variant to analyze, such as "beta".
        If None, all variants are included.

    grouping_keys:
        Columns defining an evaluation context for uniqueness checks.
        If None, the same default grouping keys as `preprocess_by_threshold`
        are used.

    enforce_unique:
        If True, enforce one row per subject per evaluation context.

    drop_subject_ids:
        Optional subject IDs to exclude from the analysis.

    subject_col:
        Subject identifier column.

    y_col:
        Binary outcome column. y=1 is interpreted as the positive class.

    label_col:
        Optional human-readable label column.

    meta_cols:
        Metadata columns to retain in the output.
        If None, defaults to ["model", "variant", "split"] when present.

    Returns
    -------
    pd.DataFrame
        One row per model / variant / split / stratum, with columns including:

            - stratum
            - score_low
            - score_high
            - n_total
            - n_stratum
            - pct_total
            - n_pos_stratum
            - n_neg_stratum
            - stratum_positive_rate
            - baseline_positive_rate
            - stratum_enrichment_factor

    Notes
    -----
    This is a stratification summary, not an enrichment-selection summary.

    Enrichment usually asks:
        "What happens if we select patients above one threshold?"

    Stratification asks:
        "How does the population composition change across multiple score bands?"
    """
    if "model" not in df.columns:
        raise KeyError("df must contain a 'model' column.")

    if score_col not in df.columns:
        raise KeyError(f"df must contain score_col={score_col!r}.")

    if y_col not in df.columns:
        raise KeyError(f"df must contain y_col={y_col!r}.")

    if subject_col not in df.columns:
        raise KeyError(f"df must contain subject_col={subject_col!r}.")

    if meta_cols is None:
        meta_cols = [c for c in ["model", "variant", "split"] if c in df.columns]

    # -------------------------
    # Resolve models
    # -------------------------
    if model is None:
        model_list = sorted(df["model"].dropna().astype(str).unique().tolist())
    elif isinstance(model, str):
        model_list = [model]
    else:
        model_list = [str(m) for m in model]

    if len(model_list) == 0:
        raise ValueError("No models selected.")

    # -------------------------
    # Validate strata
    # -------------------------
    if strata is None or len(list(strata)) == 0:
        raise ValueError("strata must contain at least one stratum.")

    strata_list: List[Stratum] = []
    for s in strata:
        if len(s) != 3:
            raise ValueError(
                "Each stratum must be a tuple: (stratum_name, score_low, score_high)."
            )

        name, low, high = s
        low = float(low)
        high = float(high)

        if low > high:
            raise ValueError(
                f"Invalid stratum {name!r}: score_low ({low}) > score_high ({high})."
            )

        strata_list.append((str(name), low, high))

    rows: List[Dict[str, Any]] = []

    # -------------------------
    # Loop over models
    # -------------------------
    for m in model_list:
        # Full eligible population for this model before stratum filtering.
        # This defines the denominator for pct_total and baseline_positive_rate.
        df_all = preprocess_by_threshold(
            df=df,
            threshold=(0.0, 1.0),
            score_col=score_col,
            split=split,
            models=m,
            variants=variants,
            grouping_keys=grouping_keys,
            enforce_unique=enforce_unique,
            drop_subject_ids=drop_subject_ids,
            subject_col=subject_col,
        )

        if df_all.empty:
            continue

        y_all = pd.to_numeric(df_all[y_col], errors="coerce").dropna().astype(int)
        n_total = int(len(y_all))
        n_pos_total = int((y_all == 1).sum())
        n_neg_total = int((y_all == 0).sum())
        baseline_positive_rate = (
            n_pos_total / n_total if n_total > 0 else float("nan")
        )

        # Metadata for the full analysis context.
        context_meta: Dict[str, Any] = {}
        for c in meta_cols:
            if c in df_all.columns:
                vals = df_all[c].dropna().unique()
                context_meta[c] = vals[0] if len(vals) == 1 else list(vals)

        # Optional label names inferred from the full population.
        pos_label = "pos"
        neg_label = "neg"
        if label_col in df_all.columns:
            label_map = _infer_label_map(df_all, y_col=y_col, label_col=label_col)
            pos_label = label_map.get(1, "pos")
            neg_label = label_map.get(0, "neg")

        # -------------------------
        # Loop over strata
        # -------------------------
        #for stratum_name, score_low, score_high in strata_list:
        for stratum_idx, (stratum_name, score_low, score_high) in enumerate(
            strata_list,
            start=1,
        ):
            df_s = preprocess_by_threshold(
                df=df,
                threshold=(score_low, score_high),
                score_col=score_col,
                split=split,
                models=m,
                variants=variants,
                grouping_keys=grouping_keys,
                enforce_unique=enforce_unique,
                drop_subject_ids=drop_subject_ids,
                subject_col=subject_col,
            )

            y_s = pd.to_numeric(df_s[y_col], errors="coerce").dropna().astype(int)

            n_stratum = int(len(y_s))
            n_pos_stratum = int((y_s == 1).sum()) if n_stratum > 0 else 0
            n_neg_stratum = int((y_s == 0).sum()) if n_stratum > 0 else 0

            stratum_positive_rate = (
                n_pos_stratum / n_stratum if n_stratum > 0 else float("nan")
            )

            stratum_negative_rate = (
                n_neg_stratum / n_stratum if n_stratum > 0 else float("nan")
            )

            pct_total = n_stratum / n_total if n_total > 0 else float("nan")

            stratum_enrichment_factor = (
                stratum_positive_rate / baseline_positive_rate
                if baseline_positive_rate and baseline_positive_rate > 0
                else float("nan")
            )
            row = {
                **context_meta,
                "score_col": score_col,
                "stratum_order": stratum_idx,
                "stratum": stratum_name,
                "score_low": score_low,
                "score_high": score_high,
                "pos_label": pos_label,
                "neg_label": neg_label,
                "n_total": n_total,
                "n_pos_total": n_pos_total,
                "n_neg_total": n_neg_total,
                "baseline_positive_rate": baseline_positive_rate,
                "n_stratum": n_stratum,
                "pct_total": pct_total,
                "n_pos_stratum": n_pos_stratum,
                "n_neg_stratum": n_neg_stratum,
                "stratum_positive_rate": stratum_positive_rate,
                "stratum_negative_rate": stratum_negative_rate,
                "stratum_enrichment_factor": stratum_enrichment_factor,
            }
    

            rows.append(row)

    if len(rows) == 0:
        raise ValueError(
            "No stratification rows were produced. Check model, split, variants, and strata."
        )

    return pd.DataFrame(rows)


def diagnostic_stratification_summary_by_model(
    df: pd.DataFrame,
    strata: Sequence[Stratum],
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a diagnostic patient-stratification summary table for one or more models.

    This function is a diagnostic-specific wrapper around the generic
    `stratification_summary_by_model(...)`.

    Interpretation
    --------------
    This wrapper is intended for diagnostic stratification, where the binary
    outcome is interpreted as:

        y = 1 -> diagnostic positive class / disease case
        y = 0 -> diagnostic negative class / control

    The score column, usually `p_mean`, is interpreted as a patient-level
    predicted probability of belonging to the diagnostic positive class.

    The function divides patients into user-defined score strata, such as:

        low diagnostic probability
        medium diagnostic probability
        high diagnostic probability

    Parameters
    ----------
    df:
        Patient-level prediction dataframe.

    strata:
        Sequence of named score intervals.

        Each stratum should be a tuple:

            (stratum_name, score_low, score_high)

        The interval is interpreted as:

            score_low <= score_col < score_high

        Any number of non-overlapping strata is allowed.

    model:
        Model name or sequence of model names to summarize.
        If None, all models in df are summarized.

    score_col:
        Patient-level predicted probability column used for stratification.

    split:
        Split to analyze, such as "test" or "external".
        If None, no split filtering is applied.

    variants:
        Prediction/calibration variant to analyze, such as "beta".
        If None, all variants are included.

    grouping_keys:
        Columns defining an evaluation context for uniqueness checks.

    enforce_unique:
        If True, enforce one row per subject per evaluation context.

    drop_subject_ids:
        Optional subject IDs to exclude from the analysis.

    subject_col:
        Subject identifier column.

    y_col:
        Binary diagnostic outcome column. y=1 is interpreted as the diagnostic
        positive class.

    label_col:
        Optional human-readable diagnostic label column.

    meta_cols:
        Metadata columns to retain in the output.

    Returns
    -------
    pd.DataFrame
        Diagnostic stratification summary table with diagnostic-specific column
        names, including:

            - baseline_diagnostic_positive_rate
            - stratum_diagnostic_positive_rate
            - stratum_diagnostic_negative_rate
            - diagnostic_enrichment_factor
            - n_diagnostic_positive_stratum
            - n_diagnostic_negative_stratum

    Notes
    -----
    This function does not change the underlying calculations. It calls the
    generic stratification function and renames columns into diagnostic
    terminology.
    """
    out = stratification_summary_by_model(
        df=df,
        strata=strata,
        model=model,
        score_col=score_col,
        split=split,
        variants=variants,
        grouping_keys=grouping_keys,
        enforce_unique=enforce_unique,
        drop_subject_ids=drop_subject_ids,
        subject_col=subject_col,
        y_col=y_col,
        label_col=label_col,
        meta_cols=meta_cols,
    )

    return rename_stratification_columns_for_diagnostic(out)


def prognostic_stratification_summary_by_model(
    df: pd.DataFrame,
    strata: Sequence[Stratum],
    *,
    model: Optional[Union[str, Sequence[str]]] = None,
    score_col: str = "p_mean",
    split: Optional[str] = "test",
    variants: Optional[Union[str, Sequence[str]]] = None,
    grouping_keys: Optional[List[str]] = None,
    enforce_unique: bool = True,
    drop_subject_ids: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    y_col: str = "y",
    label_col: str = "group_label",
    meta_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a prognostic patient-stratification summary table for treatment response.

    This function is a prognostic-specific wrapper around the generic
    `stratification_summary_by_model(...)`.

    Interpretation
    --------------
    This wrapper is intended for prognostic stratification for treatment response,
    where the binary outcome is interpreted as:

        y = 1 -> responder
        y = 0 -> non-responder

    The score column, usually `p_mean`, is interpreted as a patient-level
    predicted probability of response.

    The function divides patients into user-defined score strata, such as:

        low likelihood of response
        medium likelihood of response
        high likelihood of response

    This is prognostic stratification, not predictive treatment-effect
    stratification. It asks:

        "How does observed response rate vary across predicted-response strata?"

    It does not ask:

        "How does treatment benefit versus control vary across strata?"

    Parameters
    ----------
    df:
        Patient-level prediction dataframe.

    strata:
        Sequence of named score intervals.

        Each stratum should be a tuple:

            (stratum_name, score_low, score_high)

        The interval is interpreted as:

            score_low <= score_col < score_high

        Any number of non-overlapping strata is allowed.

    model:
        Model name or sequence of model names to summarize.
        If None, all models in df are summarized.

    score_col:
        Patient-level predicted probability of response used for stratification.

    split:
        Split to analyze, such as "test" or "external".
        If None, no split filtering is applied.

    variants:
        Prediction/calibration variant to analyze, such as "beta".
        If None, all variants are included.

    grouping_keys:
        Columns defining an evaluation context for uniqueness checks.

    enforce_unique:
        If True, enforce one row per subject per evaluation context.

    drop_subject_ids:
        Optional subject IDs to exclude from the analysis.

    subject_col:
        Subject identifier column.

    y_col:
        Binary response outcome column. y=1 is interpreted as responder.

    label_col:
        Optional human-readable response/non-response label column.

    meta_cols:
        Metadata columns to retain in the output.

    Returns
    -------
    pd.DataFrame
        Prognostic response-stratification summary table with response-specific
        column names, including:

            - baseline_response_rate
            - stratum_response_rate
            - stratum_nonresponse_rate
            - response_enrichment_factor
            - n_responders_stratum
            - n_nonresponders_stratum

    Notes
    -----
    This function does not change the underlying calculations. It calls the
    generic stratification function and renames columns into treatment-response
    terminology.
    """
    out = stratification_summary_by_model(
        df=df,
        strata=strata,
        model=model,
        score_col=score_col,
        split=split,
        variants=variants,
        grouping_keys=grouping_keys,
        enforce_unique=enforce_unique,
        drop_subject_ids=drop_subject_ids,
        subject_col=subject_col,
        y_col=y_col,
        label_col=label_col,
        meta_cols=meta_cols,
    )

    return rename_stratification_columns_for_prognostic(out)


def plot_stratification_summary_panels(
    strat_df: pd.DataFrame,
    *,
    mode: Literal["prognostic", "diagnostic", "generic"] = "prognostic",
    model_names: Optional[str | Sequence[str]] = None,
    variant: Optional[str | Sequence[str]] = None,
    split: Optional[str] = None,
    plots: Sequence[Literal["size", "rate", "enrichment"]] = ("size", "rate", "enrichment"),
    size_y: Literal["n_stratum", "pct_total"] = "n_stratum",
    figsize: tuple[float, float] = (9.0, 4.5),
    font_size: float = 12.0,
    legend_loc: str = "best",
    x_tick_rotation: int = 0,
    model_alias: Optional[Mapping[str, str]] = None,
    model_palette: Optional[Mapping[str, str]] = None,
    show_baseline: bool = True,
    baseline_color: str = "#222222",
    baseline_lw: float = 1.5,
    baseline_ls: str = "--",
    annotate_bars: bool = True,
    annotate_decimals: int = 3,
    annotate_font_size: Optional[float] = None,
    annotate_offset: float = 0.015,
    size_ylim: Optional[tuple[float, float]] = None,
    rate_ylim: Optional[tuple[float, float]] = None,
    enrichment_ylim: Optional[tuple[float, float]] = None,
) -> None:
    """
    Plot patient-stratification summary panels.

    This function is designed for outputs from:

        - stratification_summary_by_model(...)
        - diagnostic_stratification_summary_by_model(...)
        - prognostic_stratification_summary_by_model(...)

    It produces separate grouped bar charts for:

        1. Stratum size / feasibility
        2. Stratum positive or response rate
        3. Stratum enrichment factor

    Parameters
    ----------
    strat_df:
        Stratification summary dataframe.

    mode:
        Controls default column names and plot labels.

        "prognostic":
            Expects response-specific columns, such as:
                - stratum_response_rate
                - baseline_response_rate
                - response_enrichment_factor

        "diagnostic":
            Expects diagnostic-specific columns, such as:
                - stratum_diagnostic_positive_rate
                - baseline_diagnostic_positive_rate
                - diagnostic_enrichment_factor

        "generic":
            Expects generic columns, such as:
                - stratum_positive_rate
                - baseline_positive_rate
                - stratum_enrichment_factor

    model_names:
        Model name or sequence of model names to plot.
        If None, all models are included.

    variant:
        Optional variant or sequence of variants to include, such as "beta".

    split:
        Optional split to include, such as "test".

    plots:
        Which plots to produce. Options:
            - "size"
            - "rate"
            - "enrichment"

    size_y:
        Y-axis for the size / feasibility plot.

        "n_stratum":
            Plot number of patients in each stratum.

        "pct_total":
            Plot fraction of total patients in each stratum.

    figsize:
        Figure size for each plot.

    font_size:
        Base font size.

    legend_loc:
        Legend location.

    x_tick_rotation:
        Rotation for x-axis tick labels.

    model_alias:
        Optional mapping from raw model names to display labels.

    model_palette:
        Optional mapping from display model names to colors.

    show_baseline:
        If True:
            - rate plot shows baseline positive/response rate
            - enrichment plot shows baseline enrichment factor = 1.0

    baseline_color:
        Baseline line color.

    baseline_lw:
        Baseline line width.

    baseline_ls:
        Baseline line style.

    annotate_bars:
        If True, annotate bars with numeric values.

    annotate_decimals:
        Number of decimals for bar annotations.

    annotate_font_size:
        Font size for annotations. If None, uses font_size - 3 with minimum 8.

    annotate_offset:
        Vertical offset for bar annotations.

    size_ylim, rate_ylim, enrichment_ylim:
        Optional y-axis limits for each plot.

    Returns
    -------
    None
        Displays plots.
    """

    # -------------------------
    # Mode-specific columns
    # -------------------------
    if mode == "prognostic":
        rate_col = "stratum_response_rate"
        baseline_rate_col = "baseline_response_rate"
        enrichment_col = "response_enrichment_factor"
        rate_ylabel = "Response rate"
        rate_title = "Response rate by stratum"
        baseline_rate_label = "Baseline response rate"
        enrichment_ylabel = "Response enrichment factor"
        enrichment_title = "Response enrichment factor by stratum"

    elif mode == "diagnostic":
        rate_col = "stratum_diagnostic_positive_rate"
        baseline_rate_col = "baseline_diagnostic_positive_rate"
        enrichment_col = "diagnostic_enrichment_factor"
        rate_ylabel = "Diagnostic-positive rate"
        rate_title = "Diagnostic-positive rate by stratum"
        baseline_rate_label = "Baseline diagnostic-positive rate"
        enrichment_ylabel = "Diagnostic enrichment factor"
        enrichment_title = "Diagnostic enrichment factor by stratum"

    elif mode == "generic":
        rate_col = "stratum_positive_rate"
        baseline_rate_col = "baseline_positive_rate"
        enrichment_col = "stratum_enrichment_factor"
        rate_ylabel = "Positive-class rate"
        rate_title = "Positive-class rate by stratum"
        baseline_rate_label = "Baseline positive-class rate"
        enrichment_ylabel = "Enrichment factor"
        enrichment_title = "Enrichment factor by stratum"

    else:
        raise ValueError("mode must be 'prognostic', 'diagnostic', or 'generic'.")

    # -------------------------
    # Required columns
    # -------------------------
    required_cols = {
        "model",
        "stratum",
        "stratum_order",
        "n_stratum",
        "pct_total",
        rate_col,
        baseline_rate_col,
        enrichment_col,
    }

    if variant is not None:
        required_cols.add("variant")
    if split is not None:
        required_cols.add("split")

    missing = required_cols - set(strat_df.columns)
    if missing:
        raise KeyError(f"strat_df missing required columns: {sorted(missing)}")

    if size_y not in {"n_stratum", "pct_total"}:
        raise ValueError("size_y must be 'n_stratum' or 'pct_total'.")

    # -------------------------
    # Filter data
    # -------------------------
    d = strat_df.copy()

    if model_names is None:
        raw_models = d["model"].dropna().astype(str).unique().tolist()
        model_names_list = list(raw_models)
    elif isinstance(model_names, str):
        model_names_list = [model_names]
    else:
        model_names_list = [str(m) for m in model_names]

    d = d[d["model"].astype(str).isin(model_names_list)].copy()

    if d.empty:
        raise ValueError("No rows remain after model filtering.")

    if variant is not None:
        if isinstance(variant, str):
            variants_list = [variant]
        else:
            variants_list = [str(v) for v in variant]
        d = d[d["variant"].astype(str).isin(variants_list)].copy()

    if split is not None:
        d = d[d["split"].astype(str) == str(split)].copy()

    if d.empty:
        raise ValueError("No rows remain after variant/split filtering.")

    # -------------------------
    # Display labels
    # -------------------------
    if model_alias is None:
        model_alias = {}

    d["model_display"] = d["model"].astype(str).map(lambda x: model_alias.get(x, x))

    model_order_raw = [m for m in model_names_list if m in set(d["model"].astype(str))]
    model_order = [model_alias.get(m, m) for m in model_order_raw]

    if len(set(model_order)) != len(model_order):
        dupes = (
            pd.Series(model_order)
            [pd.Series(model_order).duplicated(keep=False)]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"model_alias causes duplicate model labels {dupes}. "
            "Make aliases unique or omit aliasing for colliding model names."
        )

    # Preserve user-defined stratum order from stratum_order.
    stratum_order_df = (
        d[["stratum", "stratum_order"]]
        .drop_duplicates()
        .sort_values("stratum_order")
    )
    stratum_order = stratum_order_df["stratum"].astype(str).tolist()

    d["stratum"] = pd.Categorical(
        d["stratum"].astype(str),
        categories=stratum_order,
        ordered=True,
    )

    # Palette should use display model labels.
    if model_palette is not None:
        missing_palette = [m for m in model_order if m not in model_palette]
        if missing_palette:
            raise ValueError(
                f"model_palette missing colors for display model labels: {missing_palette}"
            )

    sns.set(style="whitegrid")

    # -------------------------
    # Helpers
    # -------------------------

    def _format_value(v: float, y_col: str) -> str:
        if y_col == "n_stratum":
            return f"{v:.0f}"
        return f"{v:.{annotate_decimals}f}"

    def _annotate(ax, *, y_col: str, ylim: Optional[tuple[float, float]]) -> None:
        if not annotate_bars:
            return

        ann_fs = (
            annotate_font_size
            if annotate_font_size is not None
            else max(8, float(font_size) - 3)
        )

        max_y = 0.0
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if pd.isna(height):
                    continue

                max_y = max(max_y, float(height))
                x = bar.get_x() + bar.get_width() / 2.0
                y = float(height) + annotate_offset

                ax.text(
                    x,
                    y,
                    _format_value(float(height), y_col),
                    ha="center",
                    va="bottom",
                    fontsize=ann_fs,
                    fontweight="bold",
                )

        if ylim is None:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, max(y1, max_y + annotate_offset + 0.05))

    def _style_axes(
        ax,
        *,
        xlabel: str,
        ylabel: str,
        title: str,
        ylim: Optional[tuple[float, float]],
    ) -> None:
        ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=font_size, fontweight="bold")
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold")

        ax.tick_params(axis="both", labelsize=font_size)
        ax.tick_params(axis="x", rotation=x_tick_rotation)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.legend(
            title="",
            loc=legend_loc,
            prop={"size": font_size, "weight": "bold"},
        )

    def _plot_bar(
        *,
        y_col: str,
        ylabel: str,
        title: str,
        ylim: Optional[tuple[float, float]],
        baseline_value: Optional[float] = None,
        baseline_label: Optional[str] = None,
    ) -> None:
        plt.figure(figsize=figsize)

        ax = sns.barplot(
            data=d,
            x="stratum",
            y=y_col,
            hue="model_display",
            order=stratum_order,
            hue_order=model_order,
            palette=model_palette,
            estimator=np.mean,
            errorbar=None,
            saturation=1,
        )

        if show_baseline and baseline_value is not None:
            ax.axhline(
                float(baseline_value),
                color=baseline_color,
                linewidth=baseline_lw,
                linestyle=baseline_ls,
                label=baseline_label,
            )

        _style_axes(
            ax,
            xlabel="Stratum",
            ylabel=ylabel,
            title=title,
            ylim=ylim,
        )

        _annotate(ax, y_col=y_col, ylim=ylim)

        plt.tight_layout()
        plt.show()

    # -------------------------
    # 1. Stratum size / feasibility
    # -------------------------
    if "size" in plots:
        if size_y == "n_stratum":
            size_ylabel = "Number of patients"
            size_title = "Stratum size"
        else:
            size_ylabel = "Fraction of total population"
            size_title = "Stratum size"

        _plot_bar(
            y_col=size_y,
            ylabel=size_ylabel,
            title=size_title,
            ylim=size_ylim,
            baseline_value=None,
            baseline_label=None,
        )

    # -------------------------
    # 2. Positive / response rate by stratum
    # -------------------------
    if "rate" in plots:
        # Usually one baseline value after model/variant/split filtering.
        baseline_vals = d[baseline_rate_col].dropna().unique()
        baseline_value = float(baseline_vals[0]) if len(baseline_vals) > 0 else None

        _plot_bar(
            y_col=rate_col,
            ylabel=rate_ylabel,
            title=rate_title,
            ylim=rate_ylim,
            baseline_value=baseline_value,
            baseline_label=(
                f"{baseline_rate_label} = {baseline_value:.3f}"
                if baseline_value is not None
                else None
            ),
        )

    # -------------------------
    # 3. Enrichment factor by stratum
    # -------------------------
    if "enrichment" in plots:
        _plot_bar(
            y_col=enrichment_col,
            ylabel=enrichment_ylabel,
            title=enrichment_title,
            ylim=enrichment_ylim,
            baseline_value=1.0,
            baseline_label="No enrichment = 1.000",
        )

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# synthetic data generation for prospective enrichment workflow
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def make_balanced_semisynthetic_cohort(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    label_col: str = "y",
    n_per_class: int = 6,
    noise_scale: float | Mapping[str, float] = 0.05,
    clip_to_observed_range: bool = True,
    random_state: int | None = 42,
    id_prefix: str = "SYN",
) -> pd.DataFrame:
    """
    Create a balanced semi-synthetic cohort by:
      1) sampling real rows with replacement within each class
      2) adding small feature-wise Gaussian noise
      3) optionally clipping each feature to its observed range

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and label.
    feature_cols : Sequence[str]
        Feature columns to perturb.
    label_col : str, default="y"
        Binary class label column.
    n_per_class : int, default=6
        Number of synthetic samples to generate per class.
    noise_scale : float | Mapping[str, float], default=0.05
        If float:
            noise SD for each feature = noise_scale * feature_sd
        If mapping:
            maps feature name -> multiplier.
    clip_to_observed_range : bool, default=True
        Whether to clip perturbed feature values to observed min/max.
    random_state : int | None, default=42
        Random seed for reproducibility.
    id_prefix : str, default="SYN"
        Prefix for synthetic row IDs.

    Returns
    -------
    pd.DataFrame
        Semi-synthetic dataframe with:
          - synthetic_id
          - source_row_index
          - y
          - perturbed feature columns
          - source feature columns with suffix "_source"
    """
    # Validate required columns.
    if label_col not in df.columns:
        raise KeyError(f"label_col='{label_col}' not found in dataframe.")

    missing_features: list[str] = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    if n_per_class <= 0:
        raise ValueError("n_per_class must be >= 1.")

    # Work on a copy so the input dataframe is never modified in place.
    work: pd.DataFrame = df.copy()

    # Keep only rows with complete data for the requested features + label.
    needed_cols: list[str] = list(feature_cols) + [label_col]
    work = work.dropna(subset=needed_cols).copy()

    # Confirm binary labels are exactly {0, 1}.
    labels = sorted(work[label_col].unique().tolist())
    if set(labels) != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0, 1}} in '{label_col}', found {labels}"
        )

    rng = np.random.default_rng(random_state)

    out_parts: list[pd.DataFrame] = []
    synthetic_counter = 1

    # Generate the same number of semi-synthetic rows per class.
    for y_val in [0, 1]:
        df_y = work[work[label_col] == y_val].copy()
        if df_y.empty:
            raise ValueError(f"No rows found for {label_col}={y_val}")

        # Sample source rows with replacement.
        sampled_idx = rng.choice(df_y.index.to_numpy(), size=n_per_class, replace=True)
        sampled = (
            df_y.loc[sampled_idx]
            .copy()
            .reset_index()
            .rename(columns={"index": "source_row_index"})
        )

        # Preserve original feature values before perturbation.
        for col in feature_cols:
            sampled[f"{col}_source"] = sampled[col]

        # Add feature-wise Gaussian noise to selected features only.
        for col in feature_cols:
            feature_sd = float(work[col].std(ddof=0))

            # If the feature has zero variance or invalid variance,
            # do not add noise to it.
            if not np.isfinite(feature_sd) or feature_sd == 0:
                noise_sd = 0.0
            else:
                if isinstance(noise_scale, Mapping):
                    mult = float(noise_scale.get(col, 0.05))
                else:
                    mult = float(noise_scale)
                noise_sd = mult * feature_sd

            noise = rng.normal(loc=0.0, scale=noise_sd, size=len(sampled))
            sampled[col] = sampled[col].astype(float) + noise

            # Keep perturbed values inside the observed range if requested.
            if clip_to_observed_range:
                col_min = float(work[col].min())
                col_max = float(work[col].max())
                sampled[col] = sampled[col].clip(lower=col_min, upper=col_max)

        # Add readable synthetic IDs.
        n_here = len(sampled)
        sampled.insert(
            0,
            "synthetic_id",
            [
                f"{id_prefix}_{i:02d}"
                for i in range(synthetic_counter, synthetic_counter + n_here)
            ],
        )
        synthetic_counter += n_here

        out_parts.append(sampled)

    # Combine class-specific pieces and shuffle output rows.
    out = pd.concat(out_parts, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return out


def synthetic_data_validation(
    final_features: Mapping[str, Any],
    feature_filters: Mapping[str, Sequence[str]] | None = None,
    *,
    make_semisynthetic: bool = False,
    semisynthetic_n_per_class: int = 6,
    semisynthetic_noise_scale: float | Mapping[str, float] = 0.05,
    semisynthetic_clip_to_observed_range: bool = True,
    semisynthetic_random_state: int | None = 42,
    semisynthetic_id_prefix: str = "SYN",
    label_col: str = "y",
) -> dict[str, dict[str, Any]]:
    """
    Make synthetic data to test validation pipeline. Restructure `final_features` into a per-model dictionary containing:

    - X as a NumPy array
    - feature names as a list
    - y as a NumPy array
    - df as a pandas DataFrame containing the selected feature columns plus y
    - optionally, a balanced semi-synthetic cohort in `df_semisynthetic`

    Parameters
    ----------
    final_features : Mapping[str, Any]
        Expected structure:
        {
            "final_by_model": {
                model_name: {
                    "X": np.ndarray,
                    "feature_names_selected": list[str]
                },
                ...
            },
            "y": np.ndarray
        }

    feature_filters : Mapping[str, Sequence[str]] | None, default=None
        Optional dictionary mapping model names to feature names to keep.

    make_semisynthetic : bool, default=False
        If True, automatically generate a balanced semi-synthetic cohort
        for each model.

    semisynthetic_n_per_class : int, default=6
        Number of semi-synthetic samples per class.


    semisynthetic_noise_scale : float | Mapping[str, float], default=0.05
        Controls how much Gaussian noise is added to each selected feature
        when generating the semi-synthetic cohort.

        The noise is applied feature-wise, and the standard deviation of the
        noise for each feature is computed relative to that feature's observed
        variability in the real data.

        If a single float is provided:
            noise SD for each feature = semisynthetic_noise_scale * feature_sd

        where `feature_sd` is the standard deviation of that feature in the
        observed dataset used to generate the semi-synthetic samples.

        For example, if a feature has SD = 2.0 and
        `semisynthetic_noise_scale=0.05`, then the added noise for that
        feature will have SD = 0.1.

        If a mapping is provided:
            each feature can use its own multiplier, e.g.
            {
                "hurstExp_E37": 0.03,
                "katzFD_E45": 0.08
            }

        In that case, unspecified features fall back to 0.05 inside
        `make_balanced_semisynthetic_cohort`.

        Smaller values produce synthetic samples that stay closer to the
        original sampled rows, while larger values produce more perturbed
        and more diverse synthetic samples.

    semisynthetic_clip_to_observed_range : bool, default=True
        Whether to clip perturbed feature values to the observed range.

    semisynthetic_random_state : int | None, default=42
        Random seed for reproducibility.

    semisynthetic_id_prefix : str, default="SYN"
        Prefix for synthetic sample IDs.

    label_col : str, default="y"
        Name of the target column in returned dataframes.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model dictionary containing:
        {
            "X": np.ndarray,
            "feature_names": list[str],
            "y": np.ndarray,
            "df": pd.DataFrame,
            "df_semisynthetic": pd.DataFrame | None
        }
    """
    # Shared label vector used for every model.
    y: np.ndarray = np.asarray(final_features["y"])

    # Container for final per-model outputs.
    model_data: dict[str, dict[str, Any]] = {}

    # Iterate over all models stored in final_features["final_by_model"].
    for model_name, model_info in final_features["final_by_model"].items():
        # Extract model-specific feature matrix and feature names.
        X: np.ndarray = np.asarray(model_info["X"])
        feature_names: list[str] = list(model_info["feature_names_selected"])

        # Build the real dataframe from X and feature names.
        df: pd.DataFrame = pd.DataFrame(X, columns=feature_names)

        # Apply optional model-specific feature filtering.
        if feature_filters is not None and model_name in feature_filters:
            requested_features: list[str] = list(feature_filters[model_name])

            missing_features: list[str] = [
                feature for feature in requested_features if feature not in df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"Model '{model_name}' is missing requested features: "
                    f"{missing_features}"
                )

            # Keep only requested columns in the specified order.
            df = df[requested_features]
            feature_names = requested_features

            # Rebuild X so it stays aligned with the filtered dataframe.
            X = df.to_numpy()

        # Add the label column to the real dataframe.
        df[label_col] = y

        # Optionally generate a semi-synthetic cohort using the same
        # feature names currently represented in `df`.
        df_semisynthetic: pd.DataFrame | None = None
        if make_semisynthetic:
            df_semisynthetic = make_balanced_semisynthetic_cohort(
                df=df,
                feature_cols=feature_names,
                label_col=label_col,
                n_per_class=semisynthetic_n_per_class,
                noise_scale=semisynthetic_noise_scale,
                clip_to_observed_range=semisynthetic_clip_to_observed_range,
                random_state=semisynthetic_random_state,
                id_prefix=f"{semisynthetic_id_prefix}_{model_name}",
            )

        # Store everything for this model.
        model_data[model_name] = {
            "X": X,
            "feature_names": feature_names,
            "y": y,
            "df": df,
            "df_semisynthetic": df_semisynthetic,
        }

    return model_data


