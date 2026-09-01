"""Part 1 standardized dataset setup helpers.

These helpers create the notebook-style dataset objects only. They do not run
train/validation splitting, preprocessing, feature selection, or model training.
"""

from __future__ import annotations

from numbers import Number
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


FEATURE_GROUP_KEYS = (
    "categorical_cols",
    "ordinal_cols",
    "categorical_passthrough_cols",
    "ordinal_passthrough_cols",
    "drop_cols",
)

DROP_NAME_HINTS = (
    "id",
    "uuid",
    "guid",
    "index",
    "record",
    "subject",
    "participant",
    "patient",
    "sample",
    "date",
    "time",
    "timestamp",
    "datetime",
    "note",
    "notes",
    "comment",
    "comments",
    "free_text",
    "freetext",
    "text",
)

ORDINAL_NAME_HINTS = (
    "stage",
    "grade",
    "class",
    "severity",
    "score",
    "level",
    "ecog",
    "nyha",
    "rank",
    "scale",
)

ORDINAL_VALUE_SETS = (
    {"low", "medium", "high"},
    {"mild", "moderate", "severe"},
    {"none", "mild", "moderate", "severe"},
    {"i", "ii", "iii", "iv"},
    {"stage i", "stage ii", "stage iii", "stage iv"},
    {"grade i", "grade ii", "grade iii", "grade iv"},
)


def build_standardized_dataset_setup(
    df: pd.DataFrame,
    target_col: str,
    positive_class_value: Any,
    dataset_name: str = "uploaded_csv",
    display_name: str = "Uploaded CSV dataset",
    source: str = "user_uploaded_csv",
    task_type: str = "binary_classification",
) -> dict[str, Any]:
    """Build the Part 1 standardized dataset setup dictionary."""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the DataFrame.")

    setup_df = df.copy()
    target_values = get_unique_non_null_target_values(setup_df, target_col)
    if len(target_values) != 2:
        raise ValueError(
            f"Expected exactly two non-null target values in '{target_col}', "
            f"found {len(target_values)}."
        )

    positive_value = match_target_value(positive_class_value, target_values)
    if positive_value is None:
        formatted_values = ", ".join(_format_value(value) for value in target_values)
        raise ValueError(
            f"Positive class value '{positive_class_value}' did not match one of: "
            f"{formatted_values}."
        )

    negative_value = next(value for value in target_values if value != positive_value)
    X = setup_df.drop(columns=[target_col])
    y = setup_df[target_col]
    feature_names = list(X.columns)
    metadata = {
        "dataset_name": dataset_name,
        "display_name": display_name,
        "source": source,
        "task_type": task_type,
        "target_name": target_col,
    }
    target_mapping = {
        negative_value: 0.0,
        positive_value: 1.0,
    }

    return {
        "df": setup_df,
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "metadata": metadata,
        "target_mapping": target_mapping,
        "target_col": target_col,
        "positive_class_value": positive_value,
        "negative_class_value": negative_value,
    }


def infer_feature_groups_for_preprocessing(
    X: pd.DataFrame,
    target_col: str | None = None,
) -> dict[str, Any]:
    """Infer feature-group lists for later preprocessing."""

    feature_groups: dict[str, Any] = {
        "categorical_cols": [],
        "ordinal_cols": [],
        "categorical_passthrough_cols": [],
        "ordinal_passthrough_cols": [],
        "drop_cols": [],
        "reasoning": {
            "summary": "",
            "by_column": {},
        },
    }

    for column in X.columns:
        column_name = str(column)
        if target_col is not None and column_name == target_col:
            continue

        series = X[column]
        group_name, reason = _infer_feature_group_for_column(column_name, series)
        if group_name is not None:
            feature_groups[group_name].append(column_name)
        feature_groups["reasoning"]["by_column"][column_name] = reason

    feature_groups["reasoning"]["summary"] = _build_feature_group_summary(feature_groups)
    return feature_groups


def get_unique_non_null_target_values(df: pd.DataFrame, target_col: str) -> list[Any]:
    """Return unique non-null target values in first-seen order."""

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the DataFrame.")

    return [_normalize_target_value(value) for value in pd.unique(df[target_col].dropna())]


def match_target_value(user_value: Any, allowed_values: list[Any]) -> Any | None:
    """Match user text to one of the original target values when possible."""

    for value in allowed_values:
        if user_value == value:
            return value

    user_text = str(user_value).strip()
    for value in allowed_values:
        if user_text == str(value).strip():
            return value

    for value in allowed_values:
        if user_text.casefold() == str(value).strip().casefold():
            return value

    user_number = _to_float(user_text)
    if user_number is None:
        return None

    for value in allowed_values:
        value_number = _to_float(value)
        if value_number is not None and value_number == user_number:
            return value

    return None


def render_part1_setup_review(setup: dict[str, Any]) -> str:
    """Render the standardized Part 1 setup review."""

    df = setup["df"]
    X = setup["X"]
    y = setup["y"]
    metadata = setup["metadata"]
    target_mapping = setup["target_mapping"]

    lines = [
        "Part 1 setup created.",
        "",
        "df:",
        f"{df.shape[0]} rows \u00d7 {df.shape[1]} columns",
        "",
        "X:",
        f"{X.shape[0]} rows \u00d7 {X.shape[1]} columns",
        "",
        "y:",
        f"{len(y)} values",
        "",
        "feature_names:",
        ", ".join(setup["feature_names"]),
        "",
        "metadata:",
    ]
    for key in ("dataset_name", "display_name", "source", "task_type", "target_name"):
        lines.append(f"{key}: {metadata[key]}")

    lines.extend(["", "target_mapping:"])
    for original_value, encoded_value in target_mapping.items():
        lines.append(f"{_format_value(original_value)} \u2192 {encoded_value:.1f}")

    lines.extend(
        [
            "",
            "Next step later:",
            "prepare_train_validation_bundles settings",
        ]
    )
    return "\n".join(lines)


def render_feature_group_review(feature_groups: dict[str, Any]) -> str:
    """Render the Part 2 feature-group inference review."""

    lines = ["Part 2 feature-group inference created.", ""]
    for key in FEATURE_GROUP_KEYS:
        lines.extend(
            [
                f"{key}:",
                _format_column_group(feature_groups[key]),
                "",
            ]
        )

    lines.extend(
        [
            "Reasoning:",
            feature_groups["reasoning"]["summary"],
            "",
            "Please review this feature-group setup. Reply yes to confirm, or describe changes.",
        ]
    )
    return "\n".join(lines)


def format_target_values(values: list[Any]) -> str:
    """Format available target values for a user prompt."""

    return ", ".join(_format_value(value) for value in values)


def _infer_feature_group_for_column(
    column_name: str,
    series: pd.Series,
) -> tuple[str | None, str]:
    normalized_name = _normalize_name(column_name)
    non_null = series.dropna()
    unique_values = pd.unique(non_null)
    unique_count = len(unique_values)

    if _is_drop_column(normalized_name, series):
        return "drop_cols", "Column name or dtype suggests an identifier, date/time, or free-text field."

    if _is_raw_ordinal_column(normalized_name, series, unique_values):
        return "ordinal_cols", "Raw categorical values appear ordered."

    if _is_raw_categorical_column(series, unique_count):
        return "categorical_cols", "Raw string/category column with low or moderate cardinality."

    if _is_binary_numeric_passthrough(series, unique_values):
        return "categorical_passthrough_cols", "Already-coded binary numeric feature."

    if _is_numeric_ordinal_passthrough(normalized_name, series, unique_count):
        return "ordinal_passthrough_cols", "Small-level numeric column with an ordered score/stage-like name."

    if is_numeric_dtype(series):
        return None, "Continuous numeric feature; no feature-group assignment needed."

    return None, "No feature-group rule matched."


def _is_drop_column(normalized_name: str, series: pd.Series) -> bool:
    if normalized_name.startswith("unnamed"):
        return True
    if _name_has_hint(normalized_name, DROP_NAME_HINTS):
        return True
    return is_datetime64_any_dtype(series)


def _is_raw_ordinal_column(
    normalized_name: str,
    series: pd.Series,
    unique_values: list[Any],
) -> bool:
    if not _is_raw_categorical_dtype(series):
        return False
    if _name_has_hint(normalized_name, ORDINAL_NAME_HINTS):
        return True

    normalized_values = {
        str(value).strip().casefold()
        for value in unique_values
        if str(value).strip()
    }
    if not normalized_values:
        return False
    return any(normalized_values.issubset(value_set) for value_set in ORDINAL_VALUE_SETS)


def _is_raw_categorical_column(series: pd.Series, unique_count: int) -> bool:
    if not _is_raw_categorical_dtype(series):
        return False
    row_count = len(series)
    moderate_cardinality_cutoff = max(20, int(row_count * 0.2))
    return unique_count <= moderate_cardinality_cutoff


def _is_binary_numeric_passthrough(series: pd.Series, unique_values: list[Any]) -> bool:
    if is_bool_dtype(series):
        return True
    if not is_numeric_dtype(series):
        return False
    if len(unique_values) != 2:
        return False
    numeric_values = {_to_float(value) for value in unique_values}
    return numeric_values == {0.0, 1.0}


def _is_numeric_ordinal_passthrough(
    normalized_name: str,
    series: pd.Series,
    unique_count: int,
) -> bool:
    if not is_numeric_dtype(series) or is_bool_dtype(series):
        return False
    if unique_count < 3 or unique_count > 10:
        return False
    return _name_has_hint(normalized_name, ORDINAL_NAME_HINTS)


def _is_raw_categorical_dtype(series: pd.Series) -> bool:
    return (
        is_object_dtype(series)
        or is_string_dtype(series)
        or is_categorical_dtype(series)
    )


def _build_feature_group_summary(feature_groups: dict[str, Any]) -> str:
    if all(not feature_groups[key] for key in FEATURE_GROUP_KEYS):
        return (
            "All non-target feature columns appear to be continuous numeric "
            "measurements, so no raw categorical, ordinal, passthrough, or "
            "drop columns were inferred."
        )

    assigned = [
        f"{key}: {len(feature_groups[key])}"
        for key in FEATURE_GROUP_KEYS
        if feature_groups[key]
    ]
    return "Inferred feature groups using conservative dtype, name, and cardinality rules. " + "; ".join(assigned) + "."


def _format_column_group(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "None"


def _normalize_name(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_")


def _name_has_hint(name: str, hints: tuple[str, ...]) -> bool:
    tokens = name.split("_")
    return any(name == hint or name.endswith(f"_{hint}") or hint in tokens for hint in hints)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _normalize_target_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        return float(value)
    return value
