"""Pure helpers for constructing the first standardized binary dataset setup."""

from __future__ import annotations

from numbers import Number
from typing import Any

import pandas as pd


def get_unique_non_null_target_values(df: pd.DataFrame, target_col: str) -> list[Any]:
    """Return the target's original non-null values in their observed order."""

    _validate_dataframe_and_target(df, target_col)
    return list(pd.unique(df[target_col].dropna()))


def match_target_value(value: Any, available_values: list[Any]) -> Any:
    """Match user input to one original target value without silently guessing."""

    if not available_values:
        raise ValueError("There are no available target values to match.")

    exact_matches = [candidate for candidate in available_values if _values_equal(value, candidate)]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError("The supplied target value is ambiguous.")

    if isinstance(value, str):
        cleaned_value = value.strip()
        text_matches = [
            candidate
            for candidate in available_values
            if isinstance(candidate, str) and candidate.strip().casefold() == cleaned_value.casefold()
        ]
        if len(text_matches) == 1:
            return text_matches[0]
        if len(text_matches) > 1:
            raise ValueError("The supplied target value is ambiguous.")

    numeric_value = _as_finite_float(value)
    if numeric_value is not None:
        numeric_matches = [
            candidate
            for candidate in available_values
            if _as_finite_float(candidate) == numeric_value
        ]
        if len(numeric_matches) == 1:
            return numeric_matches[0]
        if len(numeric_matches) > 1:
            raise ValueError("The supplied target value is ambiguous.")

    raise ValueError("The supplied target value does not match an available target value.")


def build_standardized_dataset_setup(
    df: pd.DataFrame,
    target_col: str,
    positive_class_value: Any,
    dataset_name: str = "uploaded_dataset",
    display_name: str = "Uploaded dataset",
    source: str = "user_uploaded_file",
    task_type: str = "binary_classification",
) -> dict[str, Any]:
    """Build the notebook-aligned binary dataset objects without training anything."""

    _validate_dataframe_and_target(df, target_col)
    target_values = get_unique_non_null_target_values(df, target_col)
    if len(target_values) != 2:
        raise ValueError(
            "Binary classification setup requires exactly two non-null target values."
        )

    positive_value = match_target_value(positive_class_value, target_values)
    negative_values = [
        candidate for candidate in target_values if not _values_equal(candidate, positive_value)
    ]
    if len(negative_values) != 1:
        raise ValueError("The negative target value could not be determined unambiguously.")
    negative_value = negative_values[0]

    preserved_df = df.copy(deep=True)
    X = preserved_df.drop(columns=[target_col])
    y = preserved_df[target_col]
    metadata = {
        "dataset_name": dataset_name,
        "display_name": display_name,
        "source": source,
        "task_type": task_type,
        "target_name": target_col,
    }

    return {
        "df": preserved_df,
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "metadata": metadata,
        "target_mapping": {
            negative_value: 0.0,
            positive_value: 1.0,
        },
        "target_col": target_col,
        "positive_class_value": positive_value,
        "negative_class_value": negative_value,
    }


def display_target_value(value: Any) -> str:
    """Render a value clearly for tool output without coercing its stored type."""

    return repr(value)


def _validate_dataframe_and_target(df: pd.DataFrame, target_col: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The active local table is not a pandas DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} is not present in the active table.")


def _values_equal(left: Any, right: Any) -> bool:
    try:
        comparison = left == right
    except Exception:
        return False
    try:
        return bool(comparison)
    except (TypeError, ValueError):
        return False


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, Number)):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if pd.notna(numeric_value) else None
