"""CSV profiling and draft ML setup helpers for the agent runner.

This module is intentionally independent from the reusable ``ai_framework``
package. Version 1 only inspects an uploaded CSV and builds setup metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype


TARGET_NAME_HINTS = (
    "target",
    "label",
    "class",
    "outcome",
    "response",
    "responder",
    "diagnosis",
    "disease",
    "status",
    "case",
    "control",
    "group",
    "phenotype",
    "y",
)

ID_NAME_HINTS = (
    "id",
    "uuid",
    "guid",
    "subject",
    "participant",
    "patient",
    "record",
    "sample",
    "index",
)


@dataclass(frozen=True)
class ColumnProfile:
    """Compact per-column profile used by the Dataset Explorer Agent."""

    name: str
    dtype: str
    missing_count: int
    missing_fraction: float
    unique_count: int
    likely_kind: str
    sample_values: list[str]


def load_csv_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load one uploaded CSV into a pandas DataFrame."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.name}")
    return pd.read_csv(path)


def load_and_profile_csv(csv_path: str | Path) -> dict[str, Any]:
    """Load a CSV file and return a JSON-serializable profile."""

    df = load_csv_dataframe(csv_path)
    return profile_dataframe(df, source_name=Path(csv_path).name)


def profile_dataframe(df: pd.DataFrame, source_name: str | None = None) -> dict[str, Any]:
    """Profile a DataFrame for initial ML setup exploration."""

    row_count, column_count = df.shape
    columns = [str(column) for column in df.columns]
    column_profiles = [_profile_column(df, column) for column in df.columns]
    id_like_columns = [
        item.name for item in column_profiles if _is_id_like(df[item.name], item)
    ]

    numeric_columns = [
        item.name
        for item in column_profiles
        if item.likely_kind == "numeric" and item.name not in id_like_columns
    ]
    categorical_columns = [
        item.name
        for item in column_profiles
        if item.likely_kind == "categorical" and item.name not in id_like_columns
    ]

    return {
        "source_name": source_name,
        "n_rows": int(row_count),
        "n_columns": int(column_count),
        "column_names": columns,
        "dtypes": {item.name: item.dtype for item in column_profiles},
        "missingness_by_column": {
            item.name: {
                "missing_count": item.missing_count,
                "missing_fraction": item.missing_fraction,
            }
            for item in column_profiles
        },
        "unique_counts_by_column": {
            item.name: item.unique_count for item in column_profiles
        },
        "likely_numeric_columns": numeric_columns,
        "likely_categorical_columns": categorical_columns,
        "possible_target_columns": _rank_possible_targets(
            column_profiles=column_profiles,
            id_like_columns=id_like_columns,
        ),
        "possible_id_like_columns": id_like_columns,
        "columns": [asdict(item) for item in column_profiles],
    }


def build_draft_ml_setup(
    profile: dict[str, Any],
    target_column: str,
    is_binary_classification: bool,
) -> dict[str, Any]:
    """Build the confirmed, draft-only ML setup summary."""

    resolved_target = resolve_column_name(profile, target_column)
    columns_to_drop = [
        column
        for column in profile["possible_id_like_columns"]
        if column != resolved_target
    ]
    feature_columns = [
        column
        for column in profile["column_names"]
        if column != resolved_target and column not in columns_to_drop
    ]

    return {
        "target_column": resolved_target,
        "task_type": (
            "binary_classification"
            if is_binary_classification
            else "not_confirmed_binary_classification"
        ),
        "feature_columns": feature_columns,
        "columns_to_drop": columns_to_drop,
        "no_training_run": True,
        "no_feature_selection_run": True,
    }


def render_dataset_profile_markdown(profile: dict[str, Any]) -> str:
    """Render a concise dataset profile with details below the summary."""

    lines = [
        "# Dataset profile",
        "",
        "## Summary",
        "",
        f"- Shape: {profile['n_rows']} rows x {profile['n_columns']} columns",
        f"- Likely target column: {_format_likely_target(profile)}",
        f"- Likely numeric feature columns: {len(profile['likely_numeric_columns'])}",
        f"- Likely categorical columns: {len(profile['likely_categorical_columns'])}",
        f"- Missingness: {_format_missingness_summary(profile)}",
        f"- ID-like columns: {_format_id_summary(profile)}",
        "",
        "## Detailed profile",
        "",
        "| Column | dtype | Missing | Missing fraction | Unique count | Likely kind |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    for column in profile["columns"]:
        lines.append(
            "| {name} | {dtype} | {missing_count} | {missing_fraction:.3f} | "
            "{unique_count} | {likely_kind} |".format(**column)
        )

    lines.extend(
        [
            "",
            "## Column groups",
            "",
            f"- Likely numeric columns: {_format_list(profile['likely_numeric_columns'])}",
            f"- Likely categorical columns: {_format_list(profile['likely_categorical_columns'])}",
            f"- Possible target columns: {_format_target_candidates(profile['possible_target_columns'])}",
            f"- Possible ID-like columns: {_format_list(profile['possible_id_like_columns'])}",
        ]
    )
    return "\n".join(lines)


def get_top_target_column(profile: dict[str, Any]) -> str | None:
    """Return the top-ranked target candidate, if one was detected."""

    candidates = profile.get("possible_target_columns") or []
    if not candidates:
        return None
    return candidates[0]["column"]


def render_draft_setup_markdown(setup: dict[str, Any]) -> str:
    """Render the confirmed draft ML setup."""

    return "\n".join(
        [
            "# Draft ML setup",
            "",
            f"- Target column: {setup['target_column']}",
            f"- Task type: {setup['task_type']}",
            f"- Feature columns: {_format_list(setup['feature_columns'])}",
            f"- Columns to drop: {_format_list(setup['columns_to_drop'])}",
            "",
            "No model training or feature selection has been run.",
        ]
    )


def resolve_column_name(profile: dict[str, Any], user_value: str) -> str:
    """Resolve user text to an existing column name."""

    value = user_value.strip()
    columns = profile["column_names"]
    if value in columns:
        return value

    normalized_value = value.casefold()
    matches = [column for column in columns if column.casefold() == normalized_value]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(
        f"Unknown column '{user_value}'. Available columns: {', '.join(columns)}"
    )


def parse_yes_no(value: str) -> bool | None:
    """Parse simple yes/no responses for task confirmation."""

    normalized = value.strip().casefold()
    if normalized in {"yes", "y", "true", "binary", "1"}:
        return True
    if normalized in {"no", "n", "false", "not binary", "0"}:
        return False
    return None


def is_affirmative(value: str) -> bool:
    """Return whether a user confirmed a suggested value."""

    return value.strip().casefold() in {"yes", "y", "true", "correct", "confirm", "confirmed"}


def _profile_column(df: pd.DataFrame, column: Any) -> ColumnProfile:
    series = df[column]
    missing_count = int(series.isna().sum())
    row_count = len(series)
    unique_count = int(series.nunique(dropna=True))
    sample_values = [
        str(value) for value in series.dropna().drop_duplicates().head(5).tolist()
    ]

    return ColumnProfile(
        name=str(column),
        dtype=str(series.dtype),
        missing_count=missing_count,
        missing_fraction=float(missing_count / row_count) if row_count else 0.0,
        unique_count=unique_count,
        likely_kind=_infer_column_kind(series, unique_count),
        sample_values=sample_values,
    )


def _infer_column_kind(series: pd.Series, unique_count: int) -> str:
    if is_bool_dtype(series):
        return "categorical"
    if is_numeric_dtype(series):
        low_cardinality_cutoff = max(10, min(20, int(len(series) * 0.05)))
        if unique_count <= low_cardinality_cutoff:
            return "categorical"
        return "numeric"
    return "categorical"


def _is_id_like(series: pd.Series, profile: ColumnProfile) -> bool:
    name = profile.name.strip().casefold()
    tokens = name.replace("-", "_").replace(" ", "_").split("_")
    has_name_hint = any(
        name == hint or name.endswith(f"_{hint}") or hint in tokens
        for hint in ID_NAME_HINTS
    )
    if has_name_hint:
        return True

    if len(series) == 0:
        return False

    if not (is_string_dtype(series) or is_object_dtype(series)):
        return False

    unique_fraction = profile.unique_count / len(series)
    return profile.missing_count == 0 and unique_fraction >= 0.95 and profile.unique_count > 20


def _rank_possible_targets(
    column_profiles: list[ColumnProfile],
    id_like_columns: list[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    id_like_set = set(id_like_columns)

    for item in column_profiles:
        if item.name in id_like_set:
            continue

        name = item.name.strip().casefold()
        score = 0
        reasons: list[str] = []

        if any(name == hint or hint in name for hint in TARGET_NAME_HINTS):
            score += 4
            reasons.append("name suggests target/outcome")

        if item.unique_count == 2:
            score += 3
            reasons.append("exactly two unique non-missing values")
        elif 2 < item.unique_count <= 20:
            score += 2
            reasons.append("low-cardinality values")

        if item.likely_kind == "categorical":
            score += 1
            reasons.append("categorical-like dtype/cardinality")

        if item.missing_fraction > 0.5:
            score -= 2
            reasons.append("high missingness")

        if score > 0:
            candidates.append(
                {
                    "column": item.name,
                    "score": score,
                    "unique_count": item.unique_count,
                    "missing_fraction": item.missing_fraction,
                    "sample_values": item.sample_values,
                    "reasons": reasons,
                }
            )

    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "None detected"


def _format_likely_target(profile: dict[str, Any]) -> str:
    target = get_top_target_column(profile)
    return target if target else "None detected"


def _format_missingness_summary(profile: dict[str, Any]) -> str:
    missingness = profile["missingness_by_column"]
    columns_with_missing = [
        column
        for column, values in missingness.items()
        if values["missing_count"] > 0
    ]
    total_missing = sum(values["missing_count"] for values in missingness.values())

    if total_missing == 0:
        return "no missing values detected"

    return (
        f"{total_missing} missing value(s) across "
        f"{len(columns_with_missing)} column(s)"
    )


def _format_id_summary(profile: dict[str, Any]) -> str:
    id_like_columns = profile["possible_id_like_columns"]
    if not id_like_columns:
        return "none detected"
    return f"{len(id_like_columns)} detected ({_format_list(id_like_columns)})"


def _format_target_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "None detected"
    return ", ".join(
        f"{candidate['column']} (unique={candidate['unique_count']})"
        for candidate in candidates
    )
