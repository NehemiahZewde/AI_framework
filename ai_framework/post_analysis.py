# post_analysis.py

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union, Sequence, List

Threshold = Union[float, Tuple[float, float]]


Threshold = Union[float, Tuple[float, float]]


def preprocess_by_threshold(
    df: pd.DataFrame,
    threshold: Threshold,
    score_col: str = "p_median",
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
    score_col: str = "p_median",
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
    score_col: str = "p_median",
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
    score_col: str = "p_median",
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
    score_col: str = "p_median",
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
