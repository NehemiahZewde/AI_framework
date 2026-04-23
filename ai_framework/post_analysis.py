# post_analysis.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union, Sequence, List, Mapping
import math
from statistics import NormalDist


Threshold = Union[float, Tuple[float, float]]


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
      5. computes PPV-based sample-size planning

    This is the top-level enrichment wrapper. It replaces the need to call:
      - `pocket_metrics_by_model(...)`
      - `enrichment_metrics_by_model(...)`
      - `ppv_precision_sample_size_from_summary(...)`
    separately.

    Why this is useful
    ------------------
    The enrichment workflow now has three layers:

      - Pocket summary:
          describes the selected subgroup itself
      - Operating summary:
          describes what the threshold does to the full screened population
      - Planning summary:
          estimates how many selected and screened patients are needed in the
          next study to validate PPV at the chosen threshold

    This wrapper keeps those outputs aligned and makes the workflow easier to
    use and less error-prone.

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
            1-row PPV-precision sample-size planning table
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
        # after applying the threshold (for example: how many
        # patients were selected, PPV, contamination, and
        # enrichment factor).
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
        # Step 2: Build the full eligible population before thresholding.
        # This is the denominator population used to understand what the
        # threshold is doing overall, not just inside the selected pocket.
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
        # Step 3: Compute threshold operating characteristics using the
        # selected subgroup (df_hi) and the full eligible population (df_all).
        # This tells us how the threshold behaves as a screening rule in the
        # full population, including sensitivity, specificity, NPV,
        # percent selected, screen-fail rate, and number needed to screen.
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
        # Step 4: Use the operating summary to estimate study-planning
        # quantities for the next round of enrichment. In particular, this
        # computes how many selected patients are needed to estimate PPV with
        # the desired precision, and how many total patients would need to be
        # screened to obtain them.
        #-------------------------------------------------------
        planning_summary = ppv_precision_sample_size_from_summary(
            operating_summary,
            confidence=confidence,
            precision=precision,
            ceil_n=ceil_n,
        )

        #-------------------------------------------------------
        # Step 5: Store all outputs for this model in one place so downstream
        # code can access the pocket summary, operating summary, planning
        # summary, selected subgroup, and full population without additional
        # function calls.
        #-------------------------------------------------------
        results[str(m)] = {
            "pocket_summary": pocket_summary,
            "operating_summary": operating_summary,
            "planning_summary": planning_summary,
            "df_hi": df_hi,
            "df_all": df_all,
        }

    return results

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


