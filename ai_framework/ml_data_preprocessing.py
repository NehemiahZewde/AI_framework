# MILESTONE 6 - MODULE SPLIT - VERSION 1
# Feature preprocessing, cleaning, imputation, encoding, scaling, and QC.
# Raw train/validation preparation now lives in ml_train_validation.py.

# ml_data_preprocessing.py
# ML data preprocessing functions for clinical and biomedical feature matrices.



from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import missingno as msno

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


from tqdm.auto import tqdm
import re

import ai_framework.utils as ut

from ai_framework.ml_train_validation import (
    encode_target_labels,
    make_pipeline_progress_helpers,
    prepare_dataset,
    prepare_train_validation_bundles,
)



# ---------------------------
# Helpher functions
# ---------------------------

def build_column_characteristics_report(
    X: pd.DataFrame,
    *,
    max_values_to_show: int = 8,
    decimal_precision: int = 6,
) -> pd.DataFrame:
    """
    Build a concise factual report describing each feature column.

    The report provides objective information that an agent or user can use
    to determine whether each feature should be treated as:

    - numeric
    - categorical
    - ordinal
    - categorical passthrough
    - ordinal passthrough
    - excluded from modeling

    This function does not assign feature roles and does not return
    recommendations or rule-based flags.

    Parameters
    ----------
    X:
        Feature DataFrame to inspect. The target column should normally be
        excluded before calling this function.

    max_values_to_show:
        Maximum number of distinct observed values to display for each
        feature. If the column contains more distinct values, an ellipsis is
        added to the displayed value string.

    decimal_precision:
        Number of decimal places used for percentages and numeric range
        values in the returned report.

    Returns
    -------
    pd.DataFrame
        One row per feature with objective column characteristics.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.

    ValueError
        If X contains duplicate column names or an invalid configuration
        value is supplied.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame. "
            f"Received {type(X).__name__!r}."
        )

    if X.columns.has_duplicates:
        duplicate_columns = (
            X.columns[X.columns.duplicated()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            "X contains duplicate column names: "
            f"{duplicate_columns}"
        )

    if max_values_to_show < 1:
        raise ValueError("max_values_to_show must be at least 1.")

    if decimal_precision < 0:
        raise ValueError("decimal_precision cannot be negative.")

    row_count = int(len(X))
    report_rows: list[dict[str, Any]] = []

    for column_name in X.columns:
        series = X[column_name]
        non_null_series = series.dropna()

        non_null_count = int(non_null_series.shape[0])
        missing_count = int(series.isna().sum())
        unique_count = int(non_null_series.nunique(dropna=True))

        missing_pct = (
            round(
                100.0 * missing_count / row_count,
                decimal_precision,
            )
            if row_count > 0
            else np.nan
        )

        unique_pct_non_null = (
            round(
                100.0 * unique_count / non_null_count,
                decimal_precision,
            )
            if non_null_count > 0
            else np.nan
        )

        distinct_values = non_null_series.drop_duplicates()
        displayed_values = distinct_values.iloc[
            :max_values_to_show
        ].tolist()

        observed_values = _format_observed_values(
            displayed_values,
            truncated=unique_count > max_values_to_show,
            decimal_precision=decimal_precision,
        )

        is_numeric = bool(
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )

        all_numeric_values_whole_number: bool | None = None
        minimum: Any = None
        maximum: Any = None

        if is_numeric and non_null_count > 0:
            numeric_values = pd.to_numeric(
                non_null_series,
                errors="coerce",
            )

            numeric_array = numeric_values.to_numpy(dtype=float)

            finite_values = numeric_array[
                np.isfinite(numeric_array)
            ]

            if finite_values.size == numeric_array.size:
                all_numeric_values_whole_number = bool(
                    np.all(
                        np.isclose(
                            finite_values,
                            np.round(finite_values),
                            rtol=0.0,
                            atol=1e-9,
                        )
                    )
                )
            else:
                all_numeric_values_whole_number = False

            minimum = _round_report_number(
                numeric_values.min(),
                decimal_precision,
            )
            maximum = _round_report_number(
                numeric_values.max(),
                decimal_precision,
            )

        elif (
            pd.api.types.is_datetime64_any_dtype(series)
            and non_null_count > 0
        ):
            minimum = _to_python_scalar(non_null_series.min())
            maximum = _to_python_scalar(non_null_series.max())

        report_rows.append(
            {
                "column": str(column_name),
                "dtype": str(series.dtype),
                "row_count": row_count,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "unique_pct_non_null": unique_pct_non_null,
                "observed_values": observed_values,
                "observed_values_complete": (
                    unique_count <= max_values_to_show
                ),
                "all_numeric_values_whole_number": (
                    all_numeric_values_whole_number
                ),
                "min": minimum,
                "max": maximum,
            }
        )

    report_columns = [
        "column",
        "dtype",
        "row_count",
        "non_null_count",
        "missing_count",
        "missing_pct",
        "unique_count",
        "unique_pct_non_null",
        "observed_values",
        "observed_values_complete",
        "all_numeric_values_whole_number",
        "min",
        "max",
    ]

    return pd.DataFrame(
        report_rows,
        columns=report_columns,
    )


def _format_observed_values(
    values: list[Any],
    *,
    truncated: bool,
    decimal_precision: int,
) -> str:
    """
    Convert observed values into a compact readable string.

    The returned value is intentionally a plain string rather than a nested
    list or dictionary so that it is easy to display and easy for an agent
    to interpret.
    """
    formatted_values = [
        _format_report_value(
            value,
            decimal_precision=decimal_precision,
        )
        for value in values
    ]

    if truncated:
        formatted_values.append("...")

    return ", ".join(formatted_values)


def _format_report_value(
    value: Any,
    *,
    decimal_precision: int,
) -> str:
    """Format one observed value for the report."""
    value = _to_python_scalar(value)

    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if np.isnan(value):
            return "NaN"

        if np.isposinf(value):
            return "Infinity"

        if np.isneginf(value):
            return "-Infinity"

        return f"{value:.{decimal_precision}g}"

    if isinstance(value, str):
        return repr(value)

    return str(value)


def _round_report_number(
    value: Any,
    decimal_precision: int,
) -> Any:
    """Convert and round a numeric summary value."""
    value = _to_python_scalar(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return value

        return round(value, decimal_precision)

    return value


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy and pandas scalar values to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    return value



def merge_subject_df_into_bundle(
    bundle: Dict[str, Any],
    df: pd.DataFrame,
    *,
    uuid_col: str = "UUID",
    bundle_uuid_in_group_id_to_key_index: int = 1,  # group_id_to_key[gid] == (label, UUID)
    keep_df_cols: Optional[list] = None,
    how: str = "left",
    store_key: str = "subject_table",
) -> Dict[str, Any]:
    """
    SUBJECT-level merge:
      - Build a bundle subject index table: group_id, label, uuid (from group_id_to_key)
      - Merge df (keyed by uuid_col) onto it using UUID
      - Store merged table into bundle[store_key] and stats into bundle[f"{store_key}__meta"]
    """
    if "group_id_to_key" not in bundle:
        raise KeyError("bundle must contain 'group_id_to_key'")

    # 1) Clean/standardize UUID in df
    df_in = df.copy()
    if keep_df_cols is not None:
        df_in = df_in[keep_df_cols].copy()

    if uuid_col not in df_in.columns:
        raise KeyError(f"df must contain uuid_col='{uuid_col}'")

    df_in[uuid_col] = df_in[uuid_col].astype(str).str.strip()

    # If duplicates in df by UUID, keep the first (policy; change if desired)
    df_in = df_in.drop_duplicates(subset=[uuid_col], keep="first")

    # UUID set for match stats
    df_uuid_set = set(df_in[uuid_col].tolist())

    # 2) Build bundle subject index: group_id -> (label, uuid)
    rows = []
    for gid, key_tuple in bundle["group_id_to_key"].items():
        label = key_tuple[0]
        uuid = key_tuple[bundle_uuid_in_group_id_to_key_index]
        rows.append({"group_id": int(gid), "label": label, "uuid": str(uuid).strip()})

    bundle_subject = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)

    # 3) Merge on UUID (rename only for the merge input)
    merged = bundle_subject.merge(
        df_in.rename(columns={uuid_col: "uuid"}),
        on="uuid",
        how=how,
        validate="1:1",
    )

    # 4) Store + bookkeeping (match = uuid existed in the df uuid set)
    matched_mask = merged["uuid"].isin(df_uuid_set)
    n_matched = int(matched_mask.sum())
    n_unmatched = int((~matched_mask).sum())

    bundle[store_key] = merged
    bundle[f"{store_key}__meta"] = {
        "uuid_col_in_df": uuid_col,
        "how": how,
        "n_groups_in_bundle": int(bundle_subject.shape[0]),
        "n_rows_in_df_after_dedup": int(df_in.shape[0]),
        "n_rows_merged": int(merged.shape[0]),
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "df_columns_merged_in": [c for c in df_in.columns if c != uuid_col],
        "dedup_policy": "drop_duplicates(keep='first') on UUID",
    }

    return bundle



def update_encoding_metadata_feature_names(
    encoding_meta: Optional[Dict[str, Any]],
    *,
    feature_name_mapping: Dict[Any, str],
) -> Optional[Dict[str, Any]]:
    """
    Update encoding metadata after output feature names have been sanitized.

    This keeps metadata aligned with the sanitized feature names used in the
    returned bundles.
    """

    # Return early if there is no metadata.
    if encoding_meta is None:
        return None

    # Copy metadata so the caller's object is not modified unexpectedly.
    meta = dict(encoding_meta)

    # Update feature_names_out if present.
    if "feature_names_out" in meta:
        meta["feature_names_out_original"] = list(meta["feature_names_out"])
        meta["feature_names_out"] = [
            feature_name_mapping.get(name, name)
            for name in meta["feature_names_out"]
        ]

    # Update output_to_source keys if present.
    if "output_to_source" in meta:
        output_to_source = meta["output_to_source"]
        meta["output_to_source_original"] = output_to_source

        meta["output_to_source"] = {
            feature_name_mapping.get(out_name, out_name): detail
            for out_name, detail in output_to_source.items()
        }

    return meta



def clean_raw_feature_columns(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Sequence[Any],
    *,
    drop_duplicate_names: bool = True,
    drop_high_missing_columns: bool = False,
    max_missing_fraction: float = 0.20,
    high_missing_exempt_cols: Optional[Sequence[str]] = None,
    drop_constant_columns: bool = True,
    drop_near_constant_features: bool = True,
    near_constant_threshold: float = 0.95,
    near_constant_feature_types: Tuple[str, ...] = ("categorical", "ordinal"),
    near_constant_check_cols: Optional[Dict[str, str]] = None,
    return_metadata: bool = False,
) -> Union[
    Tuple[np.ndarray, List[str]],
    Tuple[np.ndarray, List[str], Dict[str, Any]],
]:
    """
    Clean raw feature columns before encoding and preprocessing.

    This function performs conservative raw feature-table QA using training
    data only:

    1. Converts feature names to plain Python strings.
    2. Drops duplicate column names, keeping the first occurrence.
    3. Drops high-missingness columns when the training missing fraction is
       greater than max_missing_fraction.
    4. Drops constant-value columns, because they contain no modeling signal.
    5. Drops near-constant selected feature types when one value dominates
       the column above a specified threshold.

    Important
    ---------
    This function should be fit on training data only. The retained feature
    names should then be applied to validation data. This avoids using
    validation missingness to decide which columns to drop.

    Parameters
    ----------
    X:
        Raw feature matrix as a pandas DataFrame or numpy array.

    feature_names:
        Feature names aligned with columns of X.

    drop_duplicate_names:
        If True, drop later columns that share the same feature name.

    drop_high_missing_columns:
        If True, drop columns whose training-set missing fraction is greater
        than max_missing_fraction.

    max_missing_fraction:
        Maximum allowed missing fraction in the training set.

        Example:
            max_missing_fraction=0.20 drops columns with >20% missingness.

        The rule is strictly greater than the threshold:
            missing_fraction > max_missing_fraction

    high_missing_exempt_cols:
        Optional list of columns that should not be dropped by the
        high-missingness rule, even if their missing fraction is high.

        This is useful when missingness is clinically meaningful or structural.

    drop_constant_columns:
        If True, drop columns with only one unique non-missing value.

    drop_near_constant_features:
        If True, drop selected feature types when one value accounts for at
        least near_constant_threshold of the non-missing rows.

    near_constant_threshold:
        Dominant-value fraction threshold used to identify near-constant
        features. For example, 0.95 means a feature is dropped if one value
        accounts for at least 95% of non-missing rows.

    near_constant_feature_types:
        Feature types eligible for near-constant filtering. By default, this
        includes categorical and ordinal features.

    near_constant_check_cols:
        Optional mapping from feature name to feature type.

        Example:
            {
                "Recent_Myocardial_Infarction": "categorical",
                "Zubrod_Performance_Status": "ordinal",
            }

        Only columns listed in this mapping and whose feature type is included
        in near_constant_feature_types are checked for near-constant behavior.

    return_metadata:
        If True, return metadata describing the cleaning actions.

    Returns
    -------
    X_clean:
        Cleaned feature matrix as a numpy array.

    feature_names_clean:
        Feature names aligned with X_clean.

    metadata:
        Returned only when return_metadata=True.
    """

    # Validate max-missing threshold.
    if not 0.0 <= max_missing_fraction <= 1.0:
        raise ValueError(
            "max_missing_fraction must be between 0 and 1, inclusive."
        )

    # Validate near-constant threshold.
    if not 0.0 < near_constant_threshold <= 1.0:
        raise ValueError(
            "near_constant_threshold must be greater than 0 and less than or "
            "equal to 1."
        )

    # Convert feature names to plain Python strings.
    feature_names_in = [str(name) for name in feature_names]

    # Build a DataFrame so name-based and value-based checks are consistent.
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
        X_df.columns = feature_names_in
    else:
        X_array = np.asarray(X)

        if X_array.ndim != 2:
            raise ValueError("X must be a 2D feature matrix.")

        X_df = pd.DataFrame(X_array, columns=feature_names_in)

    # Validate feature-name alignment.
    if X_df.shape[1] != len(feature_names_in):
        raise ValueError(
            f"X has {X_df.shape[1]} columns, but feature_names has "
            f"{len(feature_names_in)} entries."
        )

    # Normalize near-constant feature types.
    near_constant_feature_types = tuple(
        str(feature_type).lower()
        for feature_type in near_constant_feature_types
    )

    # Normalize near-constant check columns.
    near_constant_check_cols = {
        str(col): str(feature_type).lower()
        for col, feature_type in dict(near_constant_check_cols or {}).items()
    }

    # Normalize high-missingness exempt columns.
    high_missing_exempt_cols = [
        str(col) for col in list(high_missing_exempt_cols or [])
    ]
    high_missing_exempt_set = set(high_missing_exempt_cols)

    # Initialize metadata.
    metadata: Dict[str, Any] = {
        "drop_duplicate_names": drop_duplicate_names,
        "drop_high_missing_columns": drop_high_missing_columns,
        "max_missing_fraction": float(max_missing_fraction),
        "high_missing_operator": ">",
        "high_missing_exempt_cols": high_missing_exempt_cols,
        "drop_constant_columns": drop_constant_columns,
        "drop_near_constant_features": drop_near_constant_features,
        "near_constant_threshold": near_constant_threshold,
        "near_constant_feature_types": near_constant_feature_types,
        "near_constant_check_cols": near_constant_check_cols,
        "n_features_before": int(X_df.shape[1]),
        "feature_names_before": feature_names_in,
        "duplicate_name_columns_dropped": [],
        "high_missing_columns_dropped": [],
        "constant_columns_dropped": [],
        "near_constant_features_dropped": [],
    }

    # ------------------------------------------------------------
    # 1. Drop duplicate column names, keeping the first occurrence.
    # ------------------------------------------------------------
    if drop_duplicate_names:
        duplicated_mask = X_df.columns.duplicated(keep="first")

        if duplicated_mask.any():
            duplicated_positions = np.where(duplicated_mask)[0].tolist()

            metadata["duplicate_name_columns_dropped"] = [
                {
                    "feature_name": str(X_df.columns[pos]),
                    "dropped_position": int(pos),
                    "reason": "duplicate_feature_name_keep_first",
                }
                for pos in duplicated_positions
            ]

            X_df = X_df.loc[:, ~duplicated_mask].copy()

    # ------------------------------------------------------------
    # 2. Drop high-missingness columns.
    # ------------------------------------------------------------
    if drop_high_missing_columns:
        high_missing_cols: List[str] = []

        missing_fraction_by_col = X_df.isna().mean()

        for col in X_df.columns:
            # Allow specific columns to be protected from this rule.
            if col in high_missing_exempt_set:
                continue

            missing_fraction = float(missing_fraction_by_col[col])
            missing_count = int(X_df[col].isna().sum())
            n_rows = int(X_df.shape[0])

            # Drop only when missingness is greater than the threshold.
            # Example: with threshold 0.20, 20.1% is dropped, 20.0% is kept.
            if missing_fraction > max_missing_fraction:
                high_missing_cols.append(col)

                metadata["high_missing_columns_dropped"].append(
                    {
                        "feature_name": str(col),
                        "reason": "high_missing_fraction",
                        "missing_count": missing_count,
                        "n_rows": n_rows,
                        "missing_fraction": missing_fraction,
                        "threshold": float(max_missing_fraction),
                        "operator": ">",
                    }
                )

        if high_missing_cols:
            X_df = X_df.drop(columns=high_missing_cols).copy()

    # ------------------------------------------------------------
    # 3. Drop constant-value columns.
    # ------------------------------------------------------------
    if drop_constant_columns:
        constant_cols: List[str] = []

        for col in X_df.columns:
            # Count unique non-missing values.
            # A column with one unique value carries no modeling information.
            n_unique = X_df[col].nunique(dropna=True)

            if n_unique <= 1:
                constant_cols.append(col)

        if constant_cols:
            metadata["constant_columns_dropped"] = [
                {
                    "feature_name": str(col),
                    "reason": "constant_value_column",
                    "n_unique_non_missing": int(X_df[col].nunique(dropna=True)),
                }
                for col in constant_cols
            ]

            X_df = X_df.drop(columns=constant_cols).copy()

    # ------------------------------------------------------------
    # 4. Drop near-constant selected feature types.
    # ------------------------------------------------------------
    if drop_near_constant_features and near_constant_check_cols:
        near_constant_cols: List[str] = []

        for col, feature_type in near_constant_check_cols.items():
            # Only check columns still present after duplicate, high-missing,
            # and constant cleanup.
            if col not in X_df.columns:
                continue

            # Only check selected feature types.
            if feature_type not in near_constant_feature_types:
                continue

            # Compute value distribution over non-missing rows.
            value_counts = X_df[col].value_counts(dropna=True)

            # Skip empty columns. Constant empty-like behavior is already
            # handled by high-missing or constant-column removal.
            if value_counts.empty:
                continue

            n_non_missing = int(value_counts.sum())
            dominant_value = value_counts.index[0]
            dominant_count = int(value_counts.iloc[0])
            dominant_fraction = float(dominant_count / n_non_missing)

            if dominant_fraction >= near_constant_threshold:
                near_constant_cols.append(col)

                metadata["near_constant_features_dropped"].append(
                    {
                        "feature_name": str(col),
                        "reason": "near_constant_feature",
                        "feature_type": str(feature_type),
                        "dominant_value": dominant_value,
                        "dominant_count": dominant_count,
                        "n_non_missing": n_non_missing,
                        "dominant_fraction": dominant_fraction,
                        "threshold": near_constant_threshold,
                    }
                )

        if near_constant_cols:
            X_df = X_df.drop(columns=near_constant_cols).copy()

    # Final outputs.
    feature_names_clean = list(X_df.columns)
    X_clean = X_df.to_numpy()

    metadata["n_features_after"] = int(X_df.shape[1])
    metadata["feature_names_after"] = feature_names_clean

    metadata["n_duplicate_name_columns_dropped"] = len(
        metadata["duplicate_name_columns_dropped"]
    )
    metadata["n_high_missing_columns_dropped"] = len(
        metadata["high_missing_columns_dropped"]
    )
    metadata["n_constant_columns_dropped"] = len(
        metadata["constant_columns_dropped"]
    )
    metadata["n_near_constant_features_dropped"] = len(
        metadata["near_constant_features_dropped"]
    )

    if return_metadata:
        return X_clean, feature_names_clean, metadata

    return X_clean, feature_names_clean



def apply_high_cardinality_handling_train_validation(
    X_train_df: pd.DataFrame,
    X_validation_df: Optional[pd.DataFrame] = None,
    *,
    high_cardinality_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Apply user-specified high-cardinality feature handling to train and
    optional validation raw feature DataFrames.

    This helper does not infer which columns should be handled. The user must
    decide the columns ahead of time and pass them through high_cardinality_kwargs.

    Supported actions
    -----------------
    1. Drop selected columns.
    2. Group selected categorical columns using training-set frequency counts.
    3. Expand selected multi-label columns into binary indicator columns.

    Train-safe behavior
    -------------------
    - Grouping thresholds are learned from the training data only.
    - Multi-label vocabularies are learned from the training data only.
    - Validation is transformed using the training-learned contract.
    """

    # ------------------------------------------------------------------
    # Defaults.
    # ------------------------------------------------------------------
    defaults: Dict[str, Any] = {
        "enabled": False,

        "drop_kwargs": {
            "cols": [],
        },

        "group_kwargs": {
            "cols": [],
            "min_count": 5,
            "group_label": "Other",
            "missing_strategy": "keep_nan",
        },

        "multi_label_kwargs": {
            "cols": [],
            "sep": ",",
            "min_count": 5,
            "prefix": "has_",
            "missing_indicator": True,
        },

        "verbose": True,
    }

    cfg: Dict[str, Any] = {
        **defaults,
        **dict(high_cardinality_kwargs or {}),
    }

    # Merge nested dictionaries.
    cfg["drop_kwargs"] = {
        **defaults["drop_kwargs"],
        **dict(cfg.get("drop_kwargs") or {}),
    }

    cfg["group_kwargs"] = {
        **defaults["group_kwargs"],
        **dict(cfg.get("group_kwargs") or {}),
    }

    cfg["multi_label_kwargs"] = {
        **defaults["multi_label_kwargs"],
        **dict(cfg.get("multi_label_kwargs") or {}),
    }

    # Copy inputs.
    X_train_out = X_train_df.copy()
    X_validation_out = X_validation_df.copy() if X_validation_df is not None else None

    # Initialize metadata.
    meta: Dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", False)),
        "drop": {},
        "group": {},
        "multi_label": {},
        "n_train_features_before": int(X_train_df.shape[1]),
        "n_validation_features_before": (
            int(X_validation_df.shape[1])
            if X_validation_df is not None
            else None
        ),
    }

    # If disabled, return unchanged data.
    if not cfg.get("enabled", False):
        meta["n_train_features_after"] = int(X_train_out.shape[1])
        meta["n_validation_features_after"] = (
            int(X_validation_out.shape[1])
            if X_validation_out is not None
            else None
        )

        return X_train_out, X_validation_out, meta

    # ------------------------------------------------------------------
    # Helper: safe indicator column names for multi-label expansion.
    # ------------------------------------------------------------------
    def _safe_token(value: Any) -> str:
        """
        Convert a label value into a safe feature-name token.

        Final feature-name sanitization can still run later, but this keeps
        intermediate multi-label columns readable and stable.
        """
        token = str(value).strip()
        token = re.sub(r"[^A-Za-z0-9_]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("_")

        if token == "":
            token = "value"

        return token

    # ------------------------------------------------------------------
    # 1. Drop selected high-cardinality columns.
    # ------------------------------------------------------------------
    drop_cols = list(cfg["drop_kwargs"].get("cols") or [])

    existing_drop_cols = [
        col for col in drop_cols
        if col in X_train_out.columns
    ]

    missing_drop_cols = [
        col for col in drop_cols
        if col not in X_train_out.columns
    ]

    if existing_drop_cols:
        X_train_out = X_train_out.drop(columns=existing_drop_cols)

        if X_validation_out is not None:
            validation_existing_drop_cols = [
                col for col in existing_drop_cols
                if col in X_validation_out.columns
            ]

            if validation_existing_drop_cols:
                X_validation_out = X_validation_out.drop(
                    columns=validation_existing_drop_cols
                )

    meta["drop"] = {
        "requested_cols": drop_cols,
        "dropped_cols": existing_drop_cols,
        "missing_requested_cols": missing_drop_cols,
    }

    # ------------------------------------------------------------------
    # 2. Group selected categorical columns by training frequency.
    # ------------------------------------------------------------------
    group_cols = list(cfg["group_kwargs"].get("cols") or [])
    group_min_count = int(cfg["group_kwargs"].get("min_count", 5))
    group_label = cfg["group_kwargs"].get("group_label", "Other")
    missing_strategy = cfg["group_kwargs"].get("missing_strategy", "keep_nan")

    if missing_strategy not in ("keep_nan", "as_category"):
        raise ValueError(
            "group_kwargs['missing_strategy'] must be either "
            "'keep_nan' or 'as_category'."
        )

    group_meta: Dict[str, Any] = {}

    for col in group_cols:
        if col not in X_train_out.columns:
            group_meta[col] = {
                "status": "skipped_missing_in_train",
            }
            continue

        # Count non-missing training values.
        train_counts = X_train_out[col].value_counts(dropna=True)

        # Categories kept as-is are learned from training only.
        kept_values = train_counts[train_counts >= group_min_count].index.tolist()

        def _group_series(s: pd.Series) -> pd.Series:
            s_out = s.copy()

            missing_mask = s_out.isna()

            # Group non-missing values not learned as frequent in training.
            rare_or_unseen_mask = (~missing_mask) & (~s_out.isin(kept_values))
            s_out.loc[rare_or_unseen_mask] = group_label

            # Either preserve missingness for later imputation or model it as a category.
            if missing_strategy == "as_category":
                s_out.loc[missing_mask] = "Missing"

            return s_out

        X_train_out[col] = _group_series(X_train_out[col])

        if X_validation_out is not None and col in X_validation_out.columns:
            X_validation_out[col] = _group_series(X_validation_out[col])

        group_meta[col] = {
            "status": "grouped",
            "min_count": group_min_count,
            "group_label": group_label,
            "missing_strategy": missing_strategy,
            "n_unique_before_train": int(train_counts.shape[0]),
            "kept_values": kept_values,
            "n_kept_values": len(kept_values),
            "training_value_counts": train_counts.to_dict(),
        }

    meta["group"] = group_meta

    # ------------------------------------------------------------------
    # 3. Expand selected multi-label columns.
    # ------------------------------------------------------------------
    multi_cols = list(cfg["multi_label_kwargs"].get("cols") or [])
    sep = str(cfg["multi_label_kwargs"].get("sep", ","))
    multi_min_count = int(cfg["multi_label_kwargs"].get("min_count", 5))
    prefix = str(cfg["multi_label_kwargs"].get("prefix", "has_"))
    missing_indicator = bool(
        cfg["multi_label_kwargs"].get("missing_indicator", True)
    )

    multi_meta: Dict[str, Any] = {}

    def _split_labels(value: Any) -> List[str]:
        """
        Split a multi-label cell into cleaned labels.
        """
        if pd.isna(value):
            return []

        labels = [
            label.strip()
            for label in str(value).split(sep)
        ]

        labels = [
            label for label in labels
            if label != ""
        ]

        return labels

    for col in multi_cols:
        if col not in X_train_out.columns:
            multi_meta[col] = {
                "status": "skipped_missing_in_train",
            }
            continue

        # Learn label vocabulary from training only.
        train_label_lists = X_train_out[col].apply(_split_labels)

        label_counts: Dict[str, int] = {}

        for labels in train_label_lists:
            # Use set(labels) so duplicate labels inside one cell do not double-count.
            for label in set(labels):
                label_counts[label] = label_counts.get(label, 0) + 1

        kept_labels = [
            label for label, count in label_counts.items()
            if count >= multi_min_count
        ]

        # Stable output order: descending count, then alphabetical.
        kept_labels = sorted(
            kept_labels,
            key=lambda label: (-label_counts[label], str(label)),
        )

        new_cols: List[str] = []

        def _add_multi_label_indicators(df: pd.DataFrame) -> pd.DataFrame:
            df_out = df.copy()
            label_lists = df_out[col].apply(_split_labels)

            for label in kept_labels:
                safe_label = _safe_token(label)
                new_col = f"{prefix}{col}_{safe_label}"

                df_out[new_col] = label_lists.apply(
                    lambda labels: int(label in set(labels))
                )

                if new_col not in new_cols:
                    new_cols.append(new_col)

            if missing_indicator:
                missing_col = f"{prefix}{col}_missing"
                df_out[missing_col] = df_out[col].isna().astype(int)

                if missing_col not in new_cols:
                    new_cols.append(missing_col)

            # Drop the original multi-label string column after expansion.
            df_out = df_out.drop(columns=[col])

            return df_out

        X_train_out = _add_multi_label_indicators(X_train_out)

        if X_validation_out is not None and col in X_validation_out.columns:
            X_validation_out = _add_multi_label_indicators(X_validation_out)
        elif X_validation_out is not None:
            # If validation lacks the original multi-label column, still create
            # the expected train-learned output columns as zeros.
            for new_col in new_cols:
                if new_col not in X_validation_out.columns:
                    X_validation_out[new_col] = 0

        multi_meta[col] = {
            "status": "expanded",
            "sep": sep,
            "min_count": multi_min_count,
            "missing_indicator": missing_indicator,
            "label_counts_train": label_counts,
            "kept_labels": kept_labels,
            "n_kept_labels": len(kept_labels),
            "new_indicator_cols": new_cols,
            "original_col_dropped": True,
        }

    meta["multi_label"] = multi_meta

    # ------------------------------------------------------------------
    # Final metadata.
    # ------------------------------------------------------------------
    meta["n_train_features_after"] = int(X_train_out.shape[1])
    meta["n_validation_features_after"] = (
        int(X_validation_out.shape[1])
        if X_validation_out is not None
        else None
    )
    meta["train_feature_names_after"] = list(X_train_out.columns)
    meta["validation_feature_names_after"] = (
        list(X_validation_out.columns)
        if X_validation_out is not None
        else None
    )

    return X_train_out, X_validation_out, meta


def impute_raw_categorical_ordinal_train_validation(
    X_train_df: pd.DataFrame,
    X_validation_df: Optional[pd.DataFrame] = None,
    *,
    encoder_config: Dict[str, Any],
    raw_categorical_ordinal_impute_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Impute raw categorical and ordinal columns before encoding.

    This is train-safe:
        - Imputation values are learned from training only.
        - Validation uses the training-learned imputation values.

    This function works with raw values that may be strings, integers, floats,
    or object dtype values.

    Supported strategies:
        - "mode": fill missing with the most frequent non-missing training value.
        - "constant": fill missing with a user-provided constant.
        - "none": do not impute that feature type before encoding.
    """

    defaults: Dict[str, Any] = {
        "enabled": True,
        "categorical_strategy": "mode",
        "ordinal_strategy": "mode",
        "categorical_fill_value": None,
        "ordinal_fill_value": None,
        "verbose": True,
    }

    cfg: Dict[str, Any] = {
        **defaults,
        **dict(raw_categorical_ordinal_impute_kwargs or {}),
    }

    X_train_out = X_train_df.copy()
    X_validation_out = X_validation_df.copy() if X_validation_df is not None else None

    meta: Dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", True)),
        "categorical_strategy": cfg.get("categorical_strategy", "mode"),
        "ordinal_strategy": cfg.get("ordinal_strategy", "mode"),
        "categorical_fill_value": cfg.get("categorical_fill_value", None),
        "ordinal_fill_value": cfg.get("ordinal_fill_value", None),
        "columns": {},
    }

    if not cfg.get("enabled", True):
        meta["n_columns_considered"] = 0
        meta["n_columns_imputed"] = 0
        return X_train_out, X_validation_out, meta

    allowed_strategies = {"mode", "constant", "none"}

    categorical_strategy = str(cfg.get("categorical_strategy", "mode"))
    ordinal_strategy = str(cfg.get("ordinal_strategy", "mode"))

    if categorical_strategy not in allowed_strategies:
        raise ValueError(
            "categorical_strategy must be one of {'mode', 'constant', 'none'}."
        )

    if ordinal_strategy not in allowed_strategies:
        raise ValueError(
            "ordinal_strategy must be one of {'mode', 'constant', 'none'}."
        )

    # Columns treated as raw categorical/ordinal before encoding.
    categorical_cols = list(
        dict.fromkeys(
            list(encoder_config.get("cat_cols") or [])
            + list(encoder_config.get("categorical_passthrough_cols") or [])
        )
    )

    ordinal_cols = list(
        dict.fromkeys(
            list(encoder_config.get("ord_cols") or [])
            + list(encoder_config.get("ordinal_passthrough_cols") or [])
        )
    )

    # If a column appears in both lists, treat it as ordinal.
    ordinal_col_set = set(ordinal_cols)

    categorical_cols = [
        col for col in categorical_cols
        if col not in ordinal_col_set
    ]

    impute_plan: Dict[str, str] = {}

    for col in categorical_cols:
        if col in X_train_out.columns:
            impute_plan[col] = "categorical"

    for col in ordinal_cols:
        if col in X_train_out.columns:
            impute_plan[col] = "ordinal"

    def _get_mode_value(series: pd.Series) -> Any:
        """
        Return the first mode value from non-missing training values.
        """
        non_missing = series.dropna()

        if non_missing.empty:
            return None

        mode_values = non_missing.mode(dropna=True)

        if mode_values.empty:
            return None

        return mode_values.iloc[0]


    def _fill_missing_and_infer(series: pd.Series, fill_value: Any) -> pd.Series:
        """
        Fill missing values without using fillna.

        We avoid Series.fillna here because recent pandas versions raise a
        FutureWarning about silent downcasting on object dtype arrays.
        """
        filled = series.where(series.notna(), fill_value)

        return filled.infer_objects(copy=False)

    for col, feature_type in impute_plan.items():
        if feature_type == "categorical":
            strategy = categorical_strategy
            constant_value = cfg.get("categorical_fill_value", None)
        else:
            strategy = ordinal_strategy
            constant_value = cfg.get("ordinal_fill_value", None)

        train_missing_before = int(X_train_out[col].isna().sum())

        validation_missing_before = (
            int(X_validation_out[col].isna().sum())
            if X_validation_out is not None and col in X_validation_out.columns
            else None
        )

        if strategy == "none":
            fill_value = None
            status = "skipped_strategy_none"

        elif strategy == "mode":
            fill_value = _get_mode_value(X_train_out[col])

            if fill_value is None:
                status = "skipped_no_train_mode"
            else:
                X_train_out[col] = _fill_missing_and_infer(
                    X_train_out[col],
                    fill_value,
                )

                if X_validation_out is not None and col in X_validation_out.columns:
                    X_validation_out[col] = _fill_missing_and_infer(
                        X_validation_out[col],
                        fill_value,
                    )

                status = "imputed"

        elif strategy == "constant":
            fill_value = constant_value

            if fill_value is None:
                status = "skipped_no_constant_fill_value"
            else:
                X_train_out[col] = _fill_missing_and_infer(
                    X_train_out[col],
                    fill_value,
                )

                if X_validation_out is not None and col in X_validation_out.columns:
                    X_validation_out[col] = _fill_missing_and_infer(
                        X_validation_out[col],
                        fill_value,
                    )

                status = "imputed"

        train_missing_after = int(X_train_out[col].isna().sum())

        validation_missing_after = (
            int(X_validation_out[col].isna().sum())
            if X_validation_out is not None and col in X_validation_out.columns
            else None
        )

        meta["columns"][col] = {
            "feature_type": feature_type,
            "strategy": strategy,
            "fill_value": fill_value,
            "status": status,
            "train_missing_before": train_missing_before,
            "train_missing_after": train_missing_after,
            "validation_missing_before": validation_missing_before,
            "validation_missing_after": validation_missing_after,
        }

    meta["n_columns_considered"] = len(impute_plan)
    meta["n_columns_imputed"] = sum(
        1 for col_meta in meta["columns"].values()
        if col_meta["status"] == "imputed"
    )

    return X_train_out, X_validation_out, meta


def plot_feature_stat_raincloud_by_type(
    summary_df: pd.DataFrame,
    *,
    feature_encoding_metadata: Optional[Dict[str, Any]] = None,
    stat: str = "mean",
    figsize: tuple[float, float] = (8, 3),
    font_size: int = 12,
    show_points: bool = True,
    violin_half: Literal["full", "left", "right"] = "left",
) -> None:
    """
    Plot feature-statistic raincloud plots separately by final encoded feature type.

    Uses feature_encoding_metadata["output_to_source"].

    Groups:
        - continuous features: type == "numeric"
        - categorical features: type in {"onehot", "categorical_passthrough"}
        - ordinal features: type in {"ordinal", "ordinal_passthrough"}
    """

    if feature_encoding_metadata is None:
        plot_feature_stat_raincloud(
            summary_df=summary_df,
            stat=stat,
            title=f"All features: per-feature {stat}",
            feature_label="all features",
            show_points=show_points,
            figsize=figsize,
            font_size=font_size,
            violin_half=violin_half,
        )
        return

    output_to_source = feature_encoding_metadata.get("output_to_source", {})

    if not isinstance(output_to_source, dict) or len(output_to_source) == 0:
        print("[WARN] feature_encoding_metadata['output_to_source'] is missing or empty.")
        plot_feature_stat_raincloud(
            summary_df=summary_df,
            stat=stat,
            title=f"All features: per-feature {stat}",
            feature_label="all features",
            show_points=show_points,
            figsize=figsize,
            font_size=font_size,
            violin_half=violin_half,
        )
        return

    continuous_features = []
    categorical_features = []
    ordinal_features = []

    for feature_name, source_info in output_to_source.items():
        feature_type = str(source_info.get("type", "")).lower()

        if feature_type == "numeric":
            continuous_features.append(feature_name)

        elif feature_type in {"onehot", "categorical_passthrough"}:
            categorical_features.append(feature_name)

        elif feature_type in {"ordinal", "ordinal_passthrough"}:
            ordinal_features.append(feature_name)

    feature_groups = {
        "continuous features": continuous_features,
        "categorical features": categorical_features,
        "ordinal features": ordinal_features,
    }

    for group_label, group_features in feature_groups.items():
        available_features = [
            feature for feature in group_features
            if feature in summary_df.columns
        ]

        if len(available_features) == 0:
            print(f"[SKIP] No available {group_label}.")
            continue

        group_summary_df = summary_df.loc[:, available_features]

        plot_feature_stat_raincloud(
            summary_df=group_summary_df,
            stat=stat,
            title=f"{group_label.title()}: per-feature {stat}",
            feature_label=group_label,
            show_points=show_points,
            figsize=figsize,
            font_size=font_size,
            violin_half=violin_half,
        )

def _align_validation_to_cleaned_train_raw_features(
    X_validation_feature_df: Optional[pd.DataFrame],
    X_train_feature_df_clean: pd.DataFrame,
    raw_feature_cleaning_meta: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    Align validation raw features to the cleaned training raw-feature contract.

    This is needed because raw feature cleaning is train-fitted. The training
    set decides which raw columns are retained, then validation must be reduced
    to exactly those retained raw columns before high-cardinality handling,
    categorical/ordinal imputation, encoding, sanitization, and numeric
    preprocessing.

    Important edge case:
    If the original raw data contains duplicate feature names, selecting
    validation columns by name before removing validation duplicates can return
    duplicate columns. In pandas, data[col] returns a DataFrame instead of a
    Series when col is duplicated, which can break encoding.

    Therefore, when duplicate-name cleanup was enabled, duplicate validation
    columns are also removed using keep="first" before validation is aligned to
    the cleaned train columns.
    """

    if X_validation_feature_df is None:
        return None

    X_validation_aligned = X_validation_feature_df.copy()

    # If duplicate-name cleanup was applied to train, apply the same duplicate
    # name policy to validation before selecting columns by name.
    if raw_feature_cleaning_meta.get("drop_duplicate_names", False):
        duplicate_mask = X_validation_aligned.columns.duplicated(keep="first")

        if bool(duplicate_mask.any()):
            X_validation_aligned = X_validation_aligned.loc[:, ~duplicate_mask].copy()

    train_clean_columns = list(X_train_feature_df_clean.columns)

    missing_validation_columns = [
        col for col in train_clean_columns
        if col not in X_validation_aligned.columns
    ]

    if missing_validation_columns:
        raise ValueError(
            "Validation data is missing columns retained after train raw-feature "
            f"cleaning: {missing_validation_columns}"
        )

    # Keep exactly the cleaned train raw-feature columns and preserve order.
    X_validation_aligned = X_validation_aligned.loc[:, train_clean_columns].copy()

    # Final defensive check. After alignment, validation columns must be unique.
    duplicated_after_alignment = list(
        X_validation_aligned.columns[
            X_validation_aligned.columns.duplicated(keep=False)
        ]
    )

    if duplicated_after_alignment:
        raise ValueError(
            "Validation raw-feature alignment still contains duplicate columns: "
            f"{duplicated_after_alignment}"
        )

    return X_validation_aligned

# ---------------------------------------------------------------------
# Data preprocessing: feature encoding and target label mapping
# ---------------------------------------------------------------------

def sanitize_feature_names(
    feature_names: Sequence[Any],
    *,
    replacement: str = "_",
    allowed_pattern: str = r"A-Za-z0-9_",
    lowercase: bool = False,
    strip_replacement: bool = True,
    collapse_replacement: bool = True,
    empty_name_prefix: str = "feature",
    make_unique: bool = True,
    return_metadata: bool = False,
) -> Union[List[str], Tuple[List[str], Dict[str, Any]]]:
    """
    Sanitize feature names so they are safe for model libraries, file outputs,
    and downstream plotting code.

    This function is intentionally general. It is not tied to one dataset or
    one model library. By default, it keeps only letters, numbers, and
    underscores, and replaces everything else with `replacement`.

    Examples
    --------
    "123 score"            -> "123_score"
    "T2 LV (cubic mm)"     -> "T2_LV_cubic_mm"
    "ncorticalGREY [ml]"   -> "ncorticalGREY_ml"
    "SEX-CODE"             -> "SEX_CODE"
    "feature<1"            -> "feature_1"
    """

    # Validate replacement.
    if not isinstance(replacement, str):
        raise TypeError("replacement must be a string.")

    # Compile invalid-character pattern.
    invalid_re = re.compile(f"[^{allowed_pattern}]+")

    # Compile repeated replacement pattern when possible.
    replacement_re = None
    if replacement and collapse_replacement:
        replacement_re = re.compile(f"{re.escape(replacement)}+")

    # Initialize containers.
    sanitized_names: List[str] = []
    original_to_sanitized: Dict[Any, str] = {}
    sanitized_to_original: Dict[str, Any] = {}
    collision_counts: Dict[str, int] = {}

    # Loop over feature names.
    for i, original in enumerate(feature_names):
        # Convert original feature name to string.
        name = str(original)

        # Optionally lowercase the feature name.
        if lowercase:
            name = name.lower()

        # Replace invalid characters with the replacement string.
        clean = invalid_re.sub(replacement, name)

        # Collapse repeated replacement characters.
        if replacement_re is not None:
            clean = replacement_re.sub(replacement, clean)

        # Strip replacement characters from the beginning and end.
        if strip_replacement and replacement:
            clean = clean.strip(replacement)

        # If the name is empty after cleaning, create a fallback name.
        if clean == "":
            clean = f"{empty_name_prefix}_{i}"

        # Guarantee uniqueness after sanitization.
        if make_unique:
            base = clean

            if base in collision_counts:
                collision_counts[base] += 1
                clean = f"{base}_{collision_counts[base]}"
            else:
                collision_counts[base] = 0

        # Store sanitized name.
        sanitized_names.append(clean)

        # Store mappings.
        original_to_sanitized[original] = clean
        sanitized_to_original[clean] = original

    # Build collision report only for names that needed suffixes.
    collisions = {
        base: count
        for base, count in collision_counts.items()
        if count > 0
    }

    # Build metadata.
    metadata: Dict[str, Any] = {
        "replacement": replacement,
        "allowed_pattern": allowed_pattern,
        "lowercase": lowercase,
        "strip_replacement": strip_replacement,
        "collapse_replacement": collapse_replacement,
        "empty_name_prefix": empty_name_prefix,
        "make_unique": make_unique,
        "original_feature_names": list(feature_names),
        "sanitized_feature_names": sanitized_names,
        "original_to_sanitized": original_to_sanitized,
        "sanitized_to_original": sanitized_to_original,
        "collisions": collisions,
    }

    # Return metadata only when requested.
    return (sanitized_names, metadata) if return_metadata else sanitized_names



def append_subject_tabular_to_X_raw(
    bundle: Dict[str, Any],
    *,
    subject_table_key: str = "subject_table",
    group_col: str = "group_id",
    drop_feature_cols: Sequence[str] = ("group_id", "label", "uuid"),
    # If None, use all columns except drop_feature_cols
    feature_cols: Optional[Sequence[str]] = None,
    # Encoder function you already have (must return (X_tab_df, meta) when return_metadata=True)
    encoder_fn= None,
    # Encoder kwargs (e.g., cat_cols, ord_cols, low_card_max, ordinal_card_max, drop_first, ...)
    encoder_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Encode subject-level covariates from bundle[subject_table_key] and append them to
    bundle["X_raw"] (epoch-level) aligned by bundle["groups"] (group_id per epoch).

    Constraints satisfied:
      - DOES NOT add any new keys to bundle.
      - Only updates: bundle["X_raw"] and bundle["feature_names"].
      - Uses group_id solely for alignment; group_id/label/uuid are NOT appended as features.

    Returns
    -------
    (bundle, meta)
        meta is returned (not stored) so you can inspect feature_names_out, mappings, etc.
    """
    if encoder_fn is None:
        raise ValueError("encoder_fn must be provided (e.g., your encode_categorical_and_ordinal).")
    encoder_kwargs = dict(encoder_kwargs or {})

    # --- Validate bundle ---
    if "X_raw" not in bundle or "groups" not in bundle or "feature_names" not in bundle:
        raise KeyError("bundle must contain 'X_raw', 'groups', and 'feature_names'.")
    if subject_table_key not in bundle:
        raise KeyError(f"bundle must contain '{subject_table_key}' (subject-level table).")

    X_raw = bundle["X_raw"]
    groups = bundle["groups"]

    if X_raw.shape[0] != len(groups):
        raise ValueError(f"X_raw has {X_raw.shape[0]} rows but groups has {len(groups)} entries.")

    subject_table = bundle[subject_table_key]
    if not isinstance(subject_table, pd.DataFrame):
        raise TypeError(f"bundle['{subject_table_key}'] must be a pandas DataFrame.")

    if group_col not in subject_table.columns:
        raise KeyError(f"subject_table must contain '{group_col}' column.")

    # Ensure group_id is int-like for safe alignment with bundle["groups"]
    st = subject_table.copy()
    st[group_col] = st[group_col].astype(int)

    # --- Choose which subject-table columns to encode as features ---
    if feature_cols is None:
        feature_cols = [c for c in st.columns if c not in set(drop_feature_cols)]
    else:
        feature_cols = list(feature_cols)

    # Make sure we did not accidentally include alignment keys
    feature_cols = [c for c in feature_cols if c not in set(drop_feature_cols)]
    if not feature_cols:
        raise ValueError("No feature columns selected to encode/append.")

    # Keep group_id in a separate vector for alignment; do not encode it
    st_keys = st[[group_col]].copy()
    st_feats = st[feature_cols].copy()

    # --- Encode subject-level features ---
    # IMPORTANT: preserve row order by encoding st_feats directly (no sorting/reindexing here)
    X_tab, meta = encoder_fn(st_feats, return_metadata=True, **encoder_kwargs)

    if not isinstance(X_tab, pd.DataFrame):
        raise TypeError("encoder_fn must return a pandas DataFrame when set_output(transform='pandas').")

    # Attach group_id index for alignment (NOT a feature column)
    X_tab = X_tab.copy()
    X_tab.index = st_keys[group_col].values  # index values are group_id aligned to st order

    # If duplicate group_id rows exist (shouldn’t, but safe), keep first
    if X_tab.index.has_duplicates:
        X_tab = X_tab[~X_tab.index.duplicated(keep="first")]

    # --- Broadcast subject rows to epoch rows using groups ---
    # Reindex by epoch group ids; missing group ids become NaN rows
    epoch_gids = pd.Index(pd.Series(groups).astype(int).values, name=group_col)
    X_tab_epoch = X_tab.reindex(epoch_gids)

    # Convert to numpy float32 for concatenation
    X_tab_epoch_np = X_tab_epoch.to_numpy(dtype=np.float32, copy=False)

    # --- Append to X_raw and update feature_names ---
    X_raw_np = np.asarray(X_raw, dtype=np.float32)  # keeps existing if already float32
    X_aug = np.concatenate([X_raw_np, X_tab_epoch_np], axis=1)

    new_feature_names = list(bundle["feature_names"]) + list(meta.get("feature_names_out", X_tab.columns.tolist()))

    # Update ONLY the requested keys
    bundle["X_raw"] = X_aug
    bundle["feature_names"] = new_feature_names

    return bundle, meta


def encode_categorical_and_ordinal(
    df: pd.DataFrame,
    *,
    # Auto-bucketing defaults.
    low_card_max: int = 3,
    ordinal_card_max: int = 10,

    # Optional column overrides.
    cat_cols: Optional[List[str]] = None,
    ord_cols: Optional[List[str]] = None,
    categorical_passthrough_cols: Optional[List[str]] = None,
    ordinal_passthrough_cols: Optional[List[str]] = None,
    drop_cols: Optional[List[str]] = None,

    # Optional explicit category mappings.
    cat_categories: Optional[Dict[str, List[Any]]] = None,
    ord_categories: Optional[Dict[str, List[Any]]] = None,

    # One-hot encoding option.
    drop_first: bool = True,

    # Handling for high-cardinality non-numeric passthrough.
    allow_non_numeric_passthrough: bool = False,

    # QC behavior for validation/external data.
    strict_categories: bool = True,

    # Return metadata.
    return_metadata: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Encode feature columns in a pandas DataFrame using sklearn-compatible encoders.

    This updated version supports train-to-validation consistency by allowing
    user-provided categorical and ordinal category maps:

        cat_categories={...}
        ord_categories={...}

    When these category maps are provided, the function uses them instead of
    learning category order from the input DataFrame. This lets validation or
    external data follow the feature contract learned from training data.

    Parameters
    ----------
    df:
        Input feature DataFrame.

    low_card_max:
        Maximum number of unique values for an automatically detected
        non-numeric column to be one-hot encoded.

    ordinal_card_max:
        Maximum number of unique values for an automatically detected
        non-numeric column to be ordinal encoded.

    cat_cols:
        Columns to force into one-hot encoding.

    ord_cols:
        Columns to force into ordinal encoding.

    categorical_passthrough_cols:
        Columns to keep unchanged while marking them as categorical/discrete
        in the returned metadata. This is useful for already-coded categorical
        features, such as binary 0/1 columns, that should not be one-hot encoded
        but should also not be treated as continuous numeric features later.

    ordinal_passthrough_cols:
        Columns to keep unchanged while marking them as ordinal/discrete
        in the returned metadata. This is useful for already-coded ordinal
        clinical scores, such as EDSS, that should preserve their original
        numeric values but should not be treated as fully continuous features
        during later imputation, capping, or scaling.

    drop_cols:
        Columns to drop before encoding.

    cat_categories:
        Optional mapping from categorical column name to allowed category list.
        If provided for a column, those categories are used instead of learning
        from the current DataFrame.

    ord_categories:
        Optional mapping from ordinal column name to allowed ordered category list.
        If provided for a column, those categories are used instead of learning
        from the current DataFrame.

    drop_first:
        Whether to drop the first one-hot encoded category.

    allow_non_numeric_passthrough:
        Whether high-cardinality non-numeric columns should be passed through.
        If False, these columns are dropped.

    strict_categories:
        If True, raise a ValueError when the input data contains categorical
        or ordinal values not present in the provided category maps.

    return_metadata:
        Whether to return encoding metadata.

    Returns
    -------
    encoded:
        Encoded feature DataFrame.

    metadata:
        Returned only when return_metadata=True. Describes fitted/used encoding.

    Notes
    -----
    For training data, call this normally and let it learn categories.

    For validation/external data, call this using the training metadata through
    `encode_categorical_and_ordinal_like_train(...)`.
    """

    # Validate input type early.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Normalize optional containers.
    cat_cols = list(cat_cols or [])
    ord_cols = list(ord_cols or [])
    categorical_passthrough_cols = list(categorical_passthrough_cols or [])
    ordinal_passthrough_cols = list(ordinal_passthrough_cols or [])
    drop_cols = list(drop_cols or [])
    cat_categories = dict(cat_categories or {})
    ord_categories = dict(ord_categories or {})

    # Copy the data to avoid modifying the caller's DataFrame.
    data = df.copy()

    # Normalize pandas missing values to np.nan.
    data = data.where(pd.notna(data), np.nan)

    # Save original input feature names for metadata.
    original_feature_names = list(data.columns)

    # Drop requested columns if they exist.
    existing_drop_cols = [c for c in drop_cols if c in data.columns]
    if existing_drop_cols:
        data = data.drop(columns=existing_drop_cols)

    # # Save feature names after dropping columns.
    # feature_names_in = list(data.columns)

    # # Check that forced categorical columns exist.
    # missing_cat = [c for c in cat_cols if c not in data.columns]

    # # Check that forced ordinal columns exist.
    # missing_ord = [c for c in ord_cols if c not in data.columns]



    # Save feature names after dropping columns.
    feature_names_in = list(data.columns)

    # ------------------------------------------------------------------
    # Coerce numeric-looking columns to numeric dtype.
    # ------------------------------------------------------------------
    # Some OpenML/UCI tables load numeric columns as object dtype.
    # If we do not coerce them here, they can be mistaken for
    # high-cardinality non-numeric columns and dropped by the encoder.
    protected_non_numeric_cols = set(
        cat_cols
        + ord_cols
        + categorical_passthrough_cols
        + ordinal_passthrough_cols
    )

    numeric_coerced_cols: List[str] = []

    for col in data.columns:
        # Do not auto-coerce columns explicitly assigned to categorical,
        # ordinal, or passthrough handling.
        if col in protected_non_numeric_cols:
            continue

        # Only attempt coercion for non-numeric columns.
        if is_numeric_dtype(data[col]):
            continue

        coerced = pd.to_numeric(data[col], errors="coerce")

        # Accept coercion only if every originally non-missing value
        # was successfully converted to numeric.
        original_non_missing = data[col].notna()
        failed_coercion = original_non_missing & coerced.isna()

        if not failed_coercion.any():
            data[col] = coerced
            numeric_coerced_cols.append(col)

    # Check that forced categorical columns exist.
    missing_cat = [c for c in cat_cols if c not in data.columns]

    # Check that forced ordinal columns exist.
    missing_ord = [c for c in ord_cols if c not in data.columns]

    # Check that forced categorical passthrough columns exist.
    missing_cat_passthrough = [
        c for c in categorical_passthrough_cols
        if c not in data.columns
    ]

    # Check that forced ordinal passthrough columns exist.
    missing_ord_passthrough = [
        c for c in ordinal_passthrough_cols
        if c not in data.columns
    ]

    # Fail if required override columns are absent.
    if (
        missing_cat
        or missing_ord
        or missing_cat_passthrough
        or missing_ord_passthrough
    ):
        raise KeyError(
            f"Columns not found in df. Missing cat_cols={missing_cat}, "
            f"ord_cols={missing_ord}, "
            f"categorical_passthrough_cols={missing_cat_passthrough}, "
            f"ordinal_passthrough_cols={missing_ord_passthrough}"
        )

    # Forced columns are excluded from automatic numeric/non-numeric bucketing.
    forced = set(
        cat_cols
        + ord_cols
        + categorical_passthrough_cols
        + ordinal_passthrough_cols
    )

    # Numeric columns pass through unless explicitly forced into special handling.
    numeric_cols: List[str] = [
        c for c in data.columns
        if is_numeric_dtype(data[c]) and c not in forced
    ]

    # Non-numeric columns that are not forced are auto-bucketed by cardinality.
    non_numeric_cols: List[str] = [
        c for c in data.columns
        if c not in numeric_cols and c not in forced
    ]

    # Initialize automatic buckets.
    categorical_auto: List[str] = []
    ordinal_auto: List[str] = []
    passthrough_auto: List[str] = []
    unique_counts: Dict[str, int] = {}

    # Auto-bucket non-numeric columns.
    if non_numeric_cols:
        counts = data[non_numeric_cols].nunique(dropna=False)
        unique_counts = counts.to_dict()

        # Low-cardinality non-numeric columns become categorical.
        categorical_auto = [
            c for c in non_numeric_cols
            if counts[c] <= low_card_max
        ]

        # Medium-cardinality non-numeric columns become ordinal.
        ordinal_auto = [
            c for c in non_numeric_cols
            if low_card_max < counts[c] <= ordinal_card_max
        ]

        # High-cardinality non-numeric columns become passthrough candidates.
        passthrough_auto = [
            c for c in non_numeric_cols
            if counts[c] > ordinal_card_max
        ]

    # Final categorical columns: explicit overrides plus auto-detected columns.
    categorical_cols = list(dict.fromkeys(cat_cols + categorical_auto))

    # Final ordinal columns: explicit overrides plus auto-detected columns.
    ordinal_cols = list(dict.fromkeys(ord_cols + ordinal_auto))

    # Final categorical passthrough columns: explicit passthrough overrides only.
    categorical_passthrough_cols = list(dict.fromkeys(categorical_passthrough_cols))

    # Final ordinal passthrough columns: explicit passthrough overrides only.
    ordinal_passthrough_cols = list(dict.fromkeys(ordinal_passthrough_cols))

    # Start high-cardinality non-numeric columns as passthrough candidates.
    non_numeric_passthrough_cols = passthrough_auto

    # Track high-cardinality non-numeric columns that are dropped.
    dropped_unspecified_non_numeric: List[str] = []

    # Drop high-cardinality non-numeric columns unless passthrough is allowed.
    if non_numeric_passthrough_cols and not allow_non_numeric_passthrough:
        dropped_unspecified_non_numeric = non_numeric_passthrough_cols
        non_numeric_passthrough_cols = []

    def categories_in_appearance_order(series: pd.Series) -> List[Any]:
        """
        Return non-missing categories in first-appearance order.
        """
        vals: List[Any] = []
        seen = set()

        # Walk through values in their observed order.
        for v in series.tolist():
            # Skip missing values.
            if pd.isna(v):
                continue

            # Keep only the first appearance of each value.
            if v not in seen:
                vals.append(v)
                seen.add(v)

        return vals

    def _check_unseen_categories(
        *,
        col: str,
        allowed: Sequence[Any],
        kind: str,
    ) -> None:
        """
        Raise a QC error if the current data has categories outside `allowed`.
        """
        # Convert allowed categories to a set.
        allowed_set = set(allowed)

        # Get observed non-missing values.
        observed = [
            v for v in data[col].dropna().unique().tolist()
        ]

        # Find values not present in the training/allowed category list.
        unseen = [v for v in observed if v not in allowed_set]

        # Fail loudly if unseen categories exist.
        if unseen:
            raise ValueError(
                f"Unseen {kind} categories detected in column {col!r}: {unseen}. "
                f"Allowed categories from training are: {list(allowed)}"
            )

    # Build final categorical category mapping.
    cat_categories_final: Dict[str, List[Any]] = {}

    # Loop over categorical columns.
    for c in categorical_cols:
        # Use provided train-learned categories when available.
        if c in cat_categories and cat_categories[c] is not None:
            cat_categories_final[c] = [
                v for v in cat_categories[c]
                if not pd.isna(v)
            ]

            # Treat unseen validation categories as QC failures when strict.
            if strict_categories:
                _check_unseen_categories(
                    col=c,
                    allowed=cat_categories_final[c],
                    kind="categorical",
                )

        # Otherwise, learn categories from this DataFrame.
        else:
            cat_categories_final[c] = categories_in_appearance_order(
                data[c].astype("object")
            )

    # Build final ordinal category mapping.
    ord_categories_final: Dict[str, List[Any]] = {}

    # Loop over ordinal columns.
    for c in ordinal_cols:
        # Use provided train-learned or user-specified ordinal order.
        if c in ord_categories and ord_categories[c] is not None:
            ord_categories_final[c] = [
                v for v in ord_categories[c]
                if not pd.isna(v)
            ]

            # Treat unseen validation ordinal values as QC failures when strict.
            if strict_categories:
                _check_unseen_categories(
                    col=c,
                    allowed=ord_categories_final[c],
                    kind="ordinal",
                )

        # Otherwise, learn ordinal order from this DataFrame.
        else:
            ord_categories_final[c] = categories_in_appearance_order(
                data[c].astype("object")
            )

    # Build final categorical passthrough category mapping.
    categorical_passthrough_categories: Dict[str, List[Any]] = {}

    # Loop over categorical passthrough columns.
    for c in categorical_passthrough_cols:
        # Learn observed categorical passthrough values from this DataFrame.
        categorical_passthrough_categories[c] = categories_in_appearance_order(
            data[c].astype("object")
        )

    # Build final ordinal passthrough category mapping.
    ordinal_passthrough_categories: Dict[str, List[Any]] = {}

    # Loop over ordinal passthrough columns.
    for c in ordinal_passthrough_cols:
        # Learn observed ordinal passthrough values from this DataFrame.
        ordinal_passthrough_categories[c] = categories_in_appearance_order(
            data[c].astype("object")
        )

    # Create one-hot encoder.
    ohe = OneHotEncoder(
        categories=[cat_categories_final[c] for c in categorical_cols] if categorical_cols else "auto",
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=False,
    )

    # Create ordinal encoder with best available missing-value support.
    try:
        ord_enc = OrdinalEncoder(
            categories=[ord_categories_final[c] for c in ordinal_cols] if ordinal_cols else "auto",
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        ordinal_has_encoded_missing = True
    except TypeError:
        ord_enc = OrdinalEncoder(
            categories=[ord_categories_final[c] for c in ordinal_cols] if ordinal_cols else "auto",
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        ordinal_has_encoded_missing = False

    # Build ColumnTransformer steps.
    transformers = []

    # Add numeric passthrough block.
    if numeric_cols:
        transformers.append(("numeric", "passthrough", numeric_cols))

    # Add optional non-numeric passthrough block.
    if non_numeric_passthrough_cols:
        transformers.append(
            ("non_numeric_passthrough", "passthrough", non_numeric_passthrough_cols)
        )

    # Add categorical passthrough block.
    if categorical_passthrough_cols:
        transformers.append(
            ("categorical_passthrough", "passthrough", categorical_passthrough_cols)
        )

    # Add ordinal passthrough block.
    if ordinal_passthrough_cols:
        transformers.append(
            ("ordinal_passthrough", "passthrough", ordinal_passthrough_cols)
        )

    # Add categorical one-hot block.
    if categorical_cols:
        transformers.append(("categorical_one_hot", ohe, categorical_cols))

    # Add ordinal encoding block.
    if ordinal_cols:
        transformers.append(("ordinal", ord_enc, ordinal_cols))

    # Create sklearn preprocessor.
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Fit and transform this DataFrame.
    encoded = preprocessor.fit_transform(data)

    # Restore categorical missingness as all-NaN one-hot blocks.
    if categorical_cols:
        # Get one-hot output names from the fitted encoder.
        ohe_fnames = list(
            preprocessor
            .named_transformers_["categorical_one_hot"]
            .get_feature_names_out(categorical_cols)
        )

        # Loop over original categorical columns.
        for col in categorical_cols:
            # One-hot block columns are named as "column_level".
            prefix = f"{col}_"

            # Identify the block for this column.
            block_cols = [c for c in ohe_fnames if c.startswith(prefix)]

            # Continue if the column has no output block.
            if not block_cols:
                continue

            # Identify rows where the original value was missing.
            missing_mask = data[col].isna()

            # Set the entire one-hot block to NaN for missing source values.
            if missing_mask.any():
                encoded.loc[missing_mask, block_cols] = np.nan

    # Restore ordinal missingness to np.nan.
    if ordinal_cols:
        for col in ordinal_cols:
            # Identify rows with missing ordinal source values.
            missing_mask = data[col].isna()

            # Restore missing encoded ordinal values to np.nan.
            if missing_mask.any() and col in encoded.columns:
                encoded.loc[missing_mask, col] = np.nan

    # Restore categorical passthrough missingness to np.nan.
    if categorical_passthrough_cols:
        for col in categorical_passthrough_cols:
            # Identify rows with missing categorical passthrough source values.
            missing_mask = data[col].isna()

            # Restore missing categorical passthrough values to np.nan.
            if missing_mask.any() and col in encoded.columns:
                encoded.loc[missing_mask, col] = np.nan

    # Restore ordinal passthrough missingness to np.nan.
    if ordinal_passthrough_cols:
        for col in ordinal_passthrough_cols:
            # Identify rows with missing ordinal passthrough source values.
            missing_mask = data[col].isna()

            # Restore missing ordinal passthrough values to np.nan.
            if missing_mask.any() and col in encoded.columns:
                encoded.loc[missing_mask, col] = np.nan

    # Capture generated output columns.
    out_cols = list(encoded.columns)

    def ohe_block_for(col: str) -> List[str]:
        """
        Return one-hot output columns for a source categorical column.
        """
        prefix = f"{col}_"
        return [c for c in out_cols if c.startswith(prefix)]

    # Reorder outputs to follow source feature order.
    desired_order: List[str] = []

    # Walk through post-drop input columns in original order.
    for col in data.columns:
        # Numeric, passthrough, ordinal, and passthrough-coded outputs keep the original name.
        if (
            col in numeric_cols
            or col in non_numeric_passthrough_cols
            or col in ordinal_cols
            or col in categorical_passthrough_cols
            or col in ordinal_passthrough_cols
        ):
            if col in out_cols:
                desired_order.append(col)

        # Categorical outputs expand into one-hot block columns.
        elif col in categorical_cols:
            desired_order.extend(ohe_block_for(col))

    # Keep only columns that actually exist.
    desired_order = [c for c in desired_order if c in out_cols]

    # Apply deterministic output order.
    encoded = encoded[desired_order]

    # Build output-to-source metadata.
    output_to_source: Dict[str, Dict[str, Any]] = {}

    # Store numeric column mapping.
    for c in numeric_cols:
        if c in encoded.columns:
            output_to_source[c] = {
                "source_col": c,
                "type": "numeric",
                "detail": None,
            }

    # Store passthrough column mapping.
    for c in non_numeric_passthrough_cols:
        if c in encoded.columns:
            output_to_source[c] = {
                "source_col": c,
                "type": "passthrough",
                "detail": None,
            }

    # Store categorical passthrough column mapping.
    for c in categorical_passthrough_cols:
        if c in encoded.columns:
            output_to_source[c] = {
                "source_col": c,
                "type": "categorical_passthrough",
                "detail": {
                    "categories": categorical_passthrough_categories.get(c, []),
                },
            }

    # Store ordinal passthrough column mapping.
    for c in ordinal_passthrough_cols:
        if c in encoded.columns:
            output_to_source[c] = {
                "source_col": c,
                "type": "ordinal_passthrough",
                "detail": {
                    "categories": ordinal_passthrough_categories.get(c, []),
                },
            }

    # Store ordinal column mapping.
    for c in ordinal_cols:
        if c in encoded.columns:
            output_to_source[c] = {
                "source_col": c,
                "type": "ordinal",
                "detail": {
                    "categories": ord_categories_final.get(c, []),
                },
            }

    # Store one-hot column mapping.
    for c in categorical_cols:
        prefix = f"{c}_"
        for oc in encoded.columns:
            if oc.startswith(prefix):
                level = oc[len(prefix):]
                output_to_source[oc] = {
                    "source_col": c,
                    "type": "onehot",
                    "detail": {
                        "level": level,
                    },
                }

    # Build metadata dictionary.
    metadata: Dict[str, Any] = {
        "low_card_max": low_card_max,
        "ordinal_card_max": ordinal_card_max,
        "drop_first": drop_first,
        "strict_categories": strict_categories,
        "missing_value": "np.nan",
        "missing_categorical_output": "all-NaN block",
        "original_feature_names": original_feature_names,
        "feature_names_in": feature_names_in,
        "numeric_coerced_cols": numeric_coerced_cols,
        "numeric_passthrough_cols": numeric_cols,
        "categorical_one_hot_input_cols": categorical_cols,
        "ordinal_encoded_input_cols": ordinal_cols,
        "categorical_passthrough_input_cols": categorical_passthrough_cols,
        "ordinal_passthrough_input_cols": ordinal_passthrough_cols,
        "non_numeric_passthrough_cols": non_numeric_passthrough_cols,
        "dropped_input_cols": existing_drop_cols,
        "dropped_high_card_non_numeric": dropped_unspecified_non_numeric,
        "unique_counts_non_numeric_auto": unique_counts,
        "categorical_cols_forced": cat_cols,
        "ordinal_cols_forced": ord_cols,
        "categorical_passthrough_cols_forced": categorical_passthrough_cols,
        "ordinal_passthrough_cols_forced": ordinal_passthrough_cols,
        "categorical_cols_auto": categorical_auto,
        "ordinal_cols_auto": ordinal_auto,
        "passthrough_cols_auto": passthrough_auto,
        "categorical_categories": cat_categories_final,
        "ordinal_categories": ord_categories_final,
        "categorical_passthrough_categories": categorical_passthrough_categories,
        "ordinal_passthrough_categories": ordinal_passthrough_categories,
        "ordinal_encoder_supports_encoded_missing_value": ordinal_has_encoded_missing,
        "feature_names_out": list(encoded.columns),
        "output_to_source": output_to_source,
    }

    # Return metadata only when requested.
    return (encoded, metadata) if return_metadata else encoded


def encode_categorical_and_ordinal_like_train(
    df: pd.DataFrame,
    train_encoding_meta: Dict[str, Any],
    *,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    strict_categories: bool = True,
    return_metadata: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Encode validation or external features using the feature contract learned
    from training data.

    This helper avoids independently deciding categorical/ordinal columns or
    category order on validation data. Instead, it uses the training metadata
    returned by `encode_categorical_and_ordinal(..., return_metadata=True)`.

    Parameters
    ----------
    df:
        Validation or external feature DataFrame.

    train_encoding_meta:
        Metadata returned from the training call to encode_categorical_and_ordinal.

    encoder_kwargs:
        Optional base encoder settings. These are used for settings such as
        drop_first, low_card_max, ordinal_card_max, drop_cols, and
        allow_non_numeric_passthrough. The categorical/ordinal column lists and
        category maps are overwritten by train_encoding_meta.

    strict_categories:
        If True, fail when validation/external data contains unseen categorical
        or ordinal categories.

    return_metadata:
        Whether to return validation encoding metadata.

    Returns
    -------
    encoded_df:
        Encoded DataFrame with columns matching the training encoded output.

    metadata:
        Returned only when return_metadata=True.
    """

    # Validate input data type.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # Copy user-provided encoder kwargs.
    cfg = dict(encoder_kwargs or {})

    # Required input feature columns from training after drop_cols were applied.
    train_feature_names_in = list(train_encoding_meta.get("feature_names_in", []))

    # Required encoded output columns from training.
    train_feature_names_out = list(train_encoding_meta.get("feature_names_out", []))

    # Fail if training metadata does not contain required keys.
    if not train_feature_names_in:
        raise KeyError("train_encoding_meta is missing or has empty 'feature_names_in'.")

    if not train_feature_names_out:
        raise KeyError("train_encoding_meta is missing or has empty 'feature_names_out'.")

    # Check that validation/external data has the required source columns.
    missing_cols = [c for c in train_feature_names_in if c not in df.columns]

    # Missing source columns are a QC failure.
    if missing_cols:
        raise KeyError(
            f"New data is missing feature columns required by training encoding: "
            f"{missing_cols}"
        )

    # Force validation to use training-learned encoding structure.
    cfg.update(
        {
            "cat_cols": list(train_encoding_meta.get("categorical_one_hot_input_cols", [])),
            "ord_cols": list(train_encoding_meta.get("ordinal_encoded_input_cols", [])),
            "cat_categories": dict(train_encoding_meta.get("categorical_categories", {})),
            "ord_categories": dict(train_encoding_meta.get("ordinal_categories", {})),
            "strict_categories": strict_categories,
            "categorical_passthrough_cols": list(train_encoding_meta.get("categorical_passthrough_input_cols", [])),
            "ordinal_passthrough_cols": list(train_encoding_meta.get("ordinal_passthrough_input_cols", [])),
            "return_metadata": True,
        }
    )

    # Encode using the training feature contract.
    encoded_df, new_meta = encode_categorical_and_ordinal(df, **cfg)

    # Check exact output column match.
    if list(encoded_df.columns) != train_feature_names_out:
        raise ValueError(
            "Encoded feature columns do not match the training feature contract.\n"
            f"Expected: {train_feature_names_out}\n"
            f"Got:      {list(encoded_df.columns)}"
        )

    # Return metadata only if requested.
    return (encoded_df, new_meta) if return_metadata else encoded_df


def check_bundle_alignment_for_preprocessing(
    bundle_orig: Dict[str, Any],
    bundle: Dict[str, Any],
    preproc_key: str = "preproc",
    require_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Check whether a new bundle can safely reuse fitted preprocessing artifacts
    from an original/reference bundle.

    This function is intended to validate alignment before applying preprocessing
    learned on `bundle_orig` to `bundle`. In particular, it checks that the new
    bundle has the expected raw feature matrix, feature names, and feature order,
    and that the original bundle contains the fitted preprocessing objects stored
    under `bundle_orig[preproc_key]`.

    The function does NOT transform any data. It only performs validation and
    returns a structured PASS/FAIL result.

    Parameters
    ----------
    bundle_orig : dict
        Reference bundle that already contains fitted preprocessing artifacts.
        Expected to contain:
          - "X_raw": np.ndarray of shape (n_samples, n_features)
          - "feature_names": list[str]
          - preproc_key (default "preproc"), containing fitted preprocessing info

    bundle : dict
        New bundle to validate against `bundle_orig` before applying the fitted
        preprocessing artifacts. Expected to contain:
          - "X_raw": np.ndarray of shape (n_samples, n_features)
          - "feature_names": list[str]

    preproc_key : str, default "preproc"
        Key in `bundle_orig` where fitted preprocessing artifacts are stored.

    require_scaler : bool, default True
        If True, require that a fitted scaler exists in
        `bundle_orig[preproc_key]["scaler"]`. Set to False if you want to validate
        compatibility for a preprocessing flow that does not require scaling.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
          - "status": str
              "PASS" if all required checks succeed, otherwise "FAIL".
          - "errors": list[str]
              Validation failures that should block preprocessing transfer.
          - "warnings": list[str]
              Non-fatal issues or soft consistency concerns.

    Checks performed
    ----------------
    The function may check for:
      - required top-level keys in both bundles
      - presence of fitted preprocessing artifacts in `bundle_orig[preproc_key]`
      - consistency between `X_raw.shape[1]` and `len(feature_names)`
      - exact feature-name alignment between the fitted preprocessing space and
        the new bundle
      - exact feature ordering
      - basic consistency of stored objects such as `caps_df`, `imputer`,
        and optionally `scaler`

    Notes
    -----
    - Exact feature order matters, not just feature-name membership. Even if two
      bundles contain the same feature names, preprocessing transfer should fail
      if the column order differs.
    - This function is designed as Part 1 of a safer preprocessing-transfer
      workflow:
          1) validate alignment
          2) apply fitted preprocessing artifacts
          3) generate the transformed output (e.g. `bundle["X_scaled"]`)
    """
    errors = []
    warnings = []

    # --- Required top-level keys ---
    for name, obj in [("bundle_orig", bundle_orig), ("bundle", bundle)]:
        if "X_raw" not in obj:
            errors.append(f"{name} is missing required key: 'X_raw'")
        if "feature_names" not in obj:
            errors.append(f"{name} is missing required key: 'feature_names'")

    if preproc_key not in bundle_orig:
        errors.append(f"bundle_orig is missing required key: '{preproc_key}'")

    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    X_orig = bundle_orig["X_raw"]
    X_new = bundle["X_raw"]
    feature_names_orig = list(bundle_orig["feature_names"])
    feature_names_new = list(bundle["feature_names"])
    preproc = bundle_orig[preproc_key]

    # --- Shape checks ---
    if X_orig.shape[1] != len(feature_names_orig):
        errors.append(
            "bundle_orig mismatch: X_raw.shape[1] != len(feature_names)"
        )

    if X_new.shape[1] != len(feature_names_new):
        errors.append(
            "bundle mismatch: X_raw.shape[1] != len(feature_names)"
        )

    # --- Fitted preproc checks ---
    if "feature_names" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'feature_names'")
    if "caps_df" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'caps_df'")
    if "imputer" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'imputer'")

    if require_scaler and "scaler" not in preproc:
        errors.append(f"bundle_orig['{preproc_key}'] is missing 'scaler'")

    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": warnings}

    fitted_feature_names = list(preproc["feature_names"])

    # --- Feature count checks ---
    if len(feature_names_new) != len(fitted_feature_names):
        errors.append(
            f"Feature count mismatch: new bundle has {len(feature_names_new)} "
            f"features but fitted preproc expects {len(fitted_feature_names)}"
        )

    # --- Exact feature order check ---
    if feature_names_new != fitted_feature_names:
        if set(feature_names_new) == set(fitted_feature_names):
            errors.append(
                "Feature names match as a set, but column order differs. "
                "Bundles are not aligned."
            )
        else:
            missing_in_new = [f for f in fitted_feature_names if f not in feature_names_new]
            extra_in_new = [f for f in feature_names_new if f not in fitted_feature_names]

            if missing_in_new:
                errors.append(
                    f"New bundle is missing fitted features: {missing_in_new[:10]}"
                    + (" ..." if len(missing_in_new) > 10 else "")
                )
            if extra_in_new:
                errors.append(
                    f"New bundle has extra features not seen in fitted bundle: {extra_in_new[:10]}"
                    + (" ..." if len(extra_in_new) > 10 else "")
                )

    # --- caps_df alignment check ---
    caps_df = preproc["caps_df"]
    if list(caps_df.index) != fitted_feature_names:
        errors.append(
            f"bundle_orig['{preproc_key}']['caps_df'] index does not match fitted feature_names"
        )

    # --- scaler sanity check ---
    scaler = preproc.get("scaler", None)
    if require_scaler:
        if scaler is None:
            errors.append("Scaler is None, but require_scaler=True")
        else:
            if hasattr(scaler, "n_features_in_"):
                skipped = preproc.get("skipped_feature_names", [])
                expected_scaler_features = len(fitted_feature_names) - len(skipped)
                    
                if scaler.n_features_in_ != expected_scaler_features:
                    errors.append(
                        f"Scaler expects {scaler.n_features_in_} features; "
                        f"expected {expected_scaler_features} based on fitted features "
                        f"minus skipped non-continuous features"
                    )

    # --- imputer sanity check ---
    imputer = preproc.get("imputer", None)
    if imputer is None:
        errors.append("Imputer is None")
    else:
        if hasattr(imputer, "n_features_in_"):
            skipped = preproc.get("skipped_feature_names", [])
            expected_imputer_features = len(fitted_feature_names) - len(skipped)

            if imputer.n_features_in_ != expected_imputer_features:
                warnings.append(
                    f"Imputer expects {imputer.n_features_in_} features; "
                    f"expected approximately {expected_imputer_features} based on skipped "
                    f"non-continuous features"
                )


    status = "PASS" if not errors else "FAIL"
    return {"status": status, "errors": errors, "warnings": warnings}


def apply_preprocessing_from_bundle(
    bundle_orig: Dict[str, Any],
    bundle: Dict[str, Any],
    preproc_key: str = "preproc",
) -> Dict[str, Any]:
    """
    Apply fitted preprocessing artifacts from a reference bundle to a new bundle.

    This function reuses preprocessing objects already fit on `bundle_orig`
    (for example capping thresholds, imputers, and scaler) and applies them
    to `bundle` without refitting. It assumes the two bundles have already
    passed an alignment check and share the same feature space and column order.

    Parameters
    ----------
    bundle_orig : dict
        Reference/original bundle containing fitted preprocessing artifacts
        under `bundle_orig[preproc_key]`.

    bundle : dict
        New bundle whose raw features (`bundle["X_raw"]`) will be transformed
        using the fitted preprocessing from `bundle_orig`.

    preproc_key : str, default "preproc"
        Key in `bundle_orig` where the fitted preprocessing artifacts are stored.

    Returns
    -------
    bundle : dict
        The input `bundle`, updated in-place with:
          - `bundle["X_scaled"]`: transformed feature matrix
          - `bundle["feature_name_to_idx"]`: feature-to-column mapping
    """
    if "X_raw" not in bundle:
        raise KeyError("bundle must contain key 'X_raw'")
    if "feature_names" not in bundle:
        raise KeyError("bundle must contain key 'feature_names'")
    if preproc_key not in bundle_orig:
        raise KeyError(f"bundle_orig must contain key '{preproc_key}'")

    preproc = bundle_orig[preproc_key]
    X_raw = np.asarray(bundle["X_raw"], dtype=np.float32)
    feature_names = list(bundle["feature_names"])
    fitted_feature_names = list(preproc["feature_names"])

    if feature_names != fitted_feature_names:
        raise ValueError(
            "Feature names/order mismatch between bundle and fitted preprocessing."
        )

    bundle["feature_name_to_idx"] = {name: i for i, name in enumerate(feature_names)}

    caps_df = preproc["caps_df"]
    imputer = preproc["imputer"]
    scaler = preproc["scaler"]
    cat_ord_imputer = preproc.get("cat_ord_imputer", None)
    skipped_feature_names = preproc.get("skipped_feature_names", [])

    skipped_set = set(skipped_feature_names)
    skip_idx = [i for i, name in enumerate(feature_names) if name in skipped_set]
    cont_idx = [i for i, name in enumerate(feature_names) if name not in skipped_set]

    # Case 1: all columns were treated as continuous during fitting
    if len(skip_idx) == 0:
        lower = caps_df.loc[feature_names, "lower"].to_numpy(dtype=np.float32)
        upper = caps_df.loc[feature_names, "upper"].to_numpy(dtype=np.float32)

        X_capped = np.clip(X_raw, lower, upper).astype(np.float32, copy=False)
        X_imputed = imputer.transform(X_capped).astype(np.float32, copy=False)
        X_scaled = scaler.transform(X_imputed).astype(np.float32, copy=False)

        bundle["X_scaled"] = X_scaled
        return bundle

    # Case 2: continuous + categorical/ordinal split
    X_out = X_raw.copy()

    # Continuous subset
    feature_names_cont = [feature_names[i] for i in cont_idx]
    X_cont = X_raw[:, cont_idx]

    lower = caps_df.loc[feature_names_cont, "lower"].to_numpy(dtype=np.float32)
    upper = caps_df.loc[feature_names_cont, "upper"].to_numpy(dtype=np.float32)

    X_cont_capped = np.clip(X_cont, lower, upper).astype(np.float32, copy=False)
    X_cont_imputed = imputer.transform(X_cont_capped).astype(np.float32, copy=False)
    X_cont_scaled = scaler.transform(X_cont_imputed).astype(np.float32, copy=False)

    X_out[:, cont_idx] = X_cont_scaled

    # Categorical / ordinal subset
    if len(skip_idx) > 0 and cat_ord_imputer is not None:
        X_cat = X_out[:, skip_idx]
        X_cat_imputed = cat_ord_imputer.transform(X_cat).astype(np.float32, copy=False)
        X_out[:, skip_idx] = X_cat_imputed

    bundle["X_scaled"] = X_out.astype(np.float32, copy=False)
    return bundle


def preprocessing_transfer_pipeline(
    bundle_orig: Dict[str, Any],
    bundle: Dict[str, Any],
    preproc_key: str = "preproc",
    require_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Validate alignment between a reference bundle and a new bundle, then apply
    the fitted preprocessing artifacts from the reference bundle to the new bundle.

    This function is a wrapper around:
      1) `check_bundle_alignment_for_preprocessing(...)`
      2) `apply_preprocessing_from_bundle(...)`

    It first checks whether `bundle` is compatible with the preprocessing
    artifacts already fit on `bundle_orig`. If validation passes, it applies
    those fitted preprocessing objects to `bundle` without refitting.

    Parameters
    ----------
    bundle_orig : dict
        Reference/original bundle containing fitted preprocessing artifacts
        under `bundle_orig[preproc_key]`.

    bundle : dict
        New bundle whose raw features (`bundle["X_raw"]`) will be transformed
        using the fitted preprocessing from `bundle_orig`.

    preproc_key : str, default "preproc"
        Key in `bundle_orig` where fitted preprocessing artifacts are stored.

    require_scaler : bool, default True
        If True, require that a fitted scaler exists during validation.
        Set to False if using a preprocessing flow that does not require scaling.

    Returns
    -------
    bundle : dict
        The input `bundle`, updated in-place with transformed features
        (typically `bundle["X_scaled"]`) if validation succeeds.

    Raises
    ------
    ValueError
        If bundle alignment validation fails.
    """
    check = check_bundle_alignment_for_preprocessing(
        bundle_orig=bundle_orig,
        bundle=bundle,
        preproc_key=preproc_key,
        require_scaler=require_scaler,
    )

    if check["status"] != "PASS":
        error_msg = "Bundle alignment check failed:\n- " + "\n- ".join(check["errors"])
        if check["warnings"]:
            error_msg += "\nWarnings:\n- " + "\n- ".join(check["warnings"])
        raise ValueError(error_msg)

    bundle = apply_preprocessing_from_bundle(
        bundle_orig=bundle_orig,
        bundle=bundle,
        preproc_key=preproc_key,
    )

    return bundle



# ---------------------------------------------------------------------
# Data preprocessing pipeline
# ---------------------------------------------------------------------


def preprocess_train_validation_bundles(
    train_bundle: Dict[str, Any],
    validation_bundle: Optional[Dict[str, Any]] = None,
    *,
    raw_feature_cleaning_kwargs: Optional[Dict[str, Any]] = None,
    high_cardinality_kwargs: Optional[Dict[str, Any]] = None,
    raw_categorical_ordinal_impute_kwargs: Optional[Dict[str, Any]] = None,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    sanitize_feature_names_kwargs: Optional[Dict[str, Any]] = None,
    preprocessing_kwargs: Optional[Dict[str, Any]] = None,
    transfer_kwargs: Optional[Dict[str, Any]] = None,
    qc_kwargs: Optional[Dict[str, Any]] = None,
    save_kwargs: Optional[Dict[str, Any]] = None,
    progress_kwargs: Optional[Dict[str, Any]] = None,
    show_progress: bool = True,
    return_dataframes: bool = False,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Preprocess raw train and optional validation bundles.

    This function expects raw bundles created by prepare_train_validation_bundles(...),
    where bundle["X_raw"] is still a raw pandas DataFrame.

    Processing order
    ----------------
    1. Validate raw train/validation bundles.
    2. Clean raw feature columns using training data only.
    3. Apply user-specified high-cardinality handling.
    4. Impute raw categorical/ordinal features before encoding.
    5. Encode categorical/ordinal features using training data only.
    6. Encode validation features using the training encoding contract.
    7. Sanitize encoded feature names.
    8. Build encoded numeric bundles.
    9. Fit capping/imputation/scaling on training only.
    10. Apply fitted preprocessing to validation without refitting.
    11. Optionally run QC plots/summaries.
    12. Optionally save outputs.
    """

    # ------------------------------------------------------------------
    # Default raw-feature-cleaning settings.
    # ------------------------------------------------------------------
    raw_feature_cleaning_defaults: Dict[str, Any] = {
        "drop_duplicate_names": True,

        # Keep this default False so older notebooks do not silently change behavior.
        # Turn it on explicitly in dataset notebooks when you want a missingness cutoff.
        "drop_high_missing_columns": False,
        "max_missing_fraction": 0.20,
        "high_missing_exempt_cols": [],

        "drop_constant_columns": True,
        "drop_near_constant_features": True,
        "near_constant_threshold": 0.95,
        "near_constant_feature_types": ("categorical", "ordinal"),
    }

    raw_feature_cleaning_config: Dict[str, Any] = {
        **raw_feature_cleaning_defaults,
        **dict(raw_feature_cleaning_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default high-cardinality settings.
    # ------------------------------------------------------------------
    high_cardinality_defaults: Dict[str, Any] = {
        "enabled": False,

        "drop_kwargs": {
            "cols": [],
        },

        "group_kwargs": {
            "cols": [],
            "min_count": 5,
            "group_label": "Other",
            "missing_strategy": "keep_nan",
        },

        "multi_label_kwargs": {
            "cols": [],
            "sep": ",",
            "min_count": 5,
            "prefix": "has_",
            "missing_indicator": True,
        },

        "verbose": True,
    }

    high_cardinality_config: Dict[str, Any] = {
        **high_cardinality_defaults,
        **dict(high_cardinality_kwargs or {}),
    }

    high_cardinality_config["drop_kwargs"] = {
        **high_cardinality_defaults["drop_kwargs"],
        **dict(high_cardinality_config.get("drop_kwargs") or {}),
    }

    high_cardinality_config["group_kwargs"] = {
        **high_cardinality_defaults["group_kwargs"],
        **dict(high_cardinality_config.get("group_kwargs") or {}),
    }

    high_cardinality_config["multi_label_kwargs"] = {
        **high_cardinality_defaults["multi_label_kwargs"],
        **dict(high_cardinality_config.get("multi_label_kwargs") or {}),
    }

    # ------------------------------------------------------------------
    # Default raw categorical/ordinal imputation settings.
    # ------------------------------------------------------------------
    raw_cat_ord_impute_defaults: Dict[str, Any] = {
        "enabled": True,
        "categorical_strategy": "mode",
        "ordinal_strategy": "mode",
        "categorical_fill_value": None,
        "ordinal_fill_value": None,
        "verbose": True,
    }

    raw_cat_ord_impute_config: Dict[str, Any] = {
        **raw_cat_ord_impute_defaults,
        **dict(raw_categorical_ordinal_impute_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default encoder settings.
    # ------------------------------------------------------------------
    encoder_defaults: Dict[str, Any] = {
        "cat_cols": [],
        "ord_cols": [],
        "categorical_passthrough_cols": [],
        "ordinal_passthrough_cols": [],
        "drop_cols": None,
        "cat_categories": None,
        "ord_categories": None,
        "low_card_max": 3,
        "ordinal_card_max": 10,
        "drop_first": True,
        "allow_non_numeric_passthrough": False,
        "strict_categories": True,
    }

    encoder_config: Dict[str, Any] = {
        **encoder_defaults,
        **dict(encoder_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default feature-name sanitization settings.
    # ------------------------------------------------------------------
    sanitize_defaults: Dict[str, Any] = {
        "enabled": True,
        "replacement": "_",
        "allowed_pattern": r"A-Za-z0-9_",
        "lowercase": False,
        "strip_replacement": True,
        "collapse_replacement": True,
        "empty_name_prefix": "feature",
        "make_unique": True,
    }

    sanitize_config: Dict[str, Any] = {
        **sanitize_defaults,
        **dict(sanitize_feature_names_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default numeric preprocessing settings.
    # ------------------------------------------------------------------
    preprocessing_defaults: Dict[str, Any] = {
        "lower_q": 0.05,
        "upper_q": 0.95,
        "continuous_impute_strategy": "median",
        "categorical_impute_strategy": "mode",
        "ordinal_impute_strategy": "mode",
        "preproc_key": "preproc",
        "meta": "auto",
    }

    preprocessing_config: Dict[str, Any] = {
        **preprocessing_defaults,
        **dict(preprocessing_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default validation transfer settings.
    # ------------------------------------------------------------------
    transfer_defaults: Dict[str, Any] = {
        "require_scaler": True,
    }

    transfer_config: Dict[str, Any] = {
        **transfer_defaults,
        **dict(transfer_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default QC settings.
    # ------------------------------------------------------------------
    qc_defaults: Dict[str, Any] = {
        "run_qc": True,
        "qc_stages": ["scaled"],
        "show_qc_sections": True,
        "max_features": 20,
        "missingness_kind": "bar",
        "missingness_color": "seagreen",
        "missingness_figsize": (10, 4),
        "missingness_fontsize": 12,
        "missingness_sort": "ascending",
        "summary_stat": "mean",
        "raincloud_figsize": (8, 4),
        "raincloud_font_size": 14,
        "violin_half": "right",
    }

    qc_config: Dict[str, Any] = {
        **qc_defaults,
        **dict(qc_kwargs or {}),
    }

    qc_stages = list(qc_config.get("qc_stages", ["scaled"]))
    allowed_qc_stages = {"raw", "scaled"}

    invalid_qc_stages = [
        stage for stage in qc_stages
        if stage not in allowed_qc_stages
    ]

    if invalid_qc_stages:
        raise ValueError(
            f"Invalid qc_stages={invalid_qc_stages}. "
            f"Allowed values are: {sorted(allowed_qc_stages)}"
        )

    qc_config["qc_stages"] = qc_stages

    # ------------------------------------------------------------------
    # Default save settings.
    # ------------------------------------------------------------------
    save_defaults: Dict[str, Any] = {
        "save": False,
        "output_dir": None,
        "train_prefix": "train_bundle_preproc",
        "validation_prefix": "validation_bundle_preproc",
        "meta_prefix": "preproc_meta",
        "compress": True,
        "save_metadata": True,
    }

    save_config: Dict[str, Any] = {
        **save_defaults,
        **dict(save_kwargs or {}),
    }

    # ------------------------------------------------------------------
    # Default progress settings.
    # ------------------------------------------------------------------
    progress_defaults: Dict[str, Any] = {
        "enabled": show_progress,
        "show_output_shapes": True,
        "return_progress_log": True,
    }

    progress_config: Dict[str, Any] = {
        **progress_defaults,
        **dict(progress_kwargs or {}),
    }

    progress_enabled: bool = bool(progress_config.get("enabled", show_progress))
    show_output_shapes: bool = bool(progress_config.get("show_output_shapes", True))
    return_progress_log: bool = bool(progress_config.get("return_progress_log", True))

    (
        progress_log,
        _describe_object,
        _start_step,
        _ok_step,
        _skip_step,
        _fail_step,
    ) = make_pipeline_progress_helpers(
        progress_enabled=progress_enabled,
        show_output_shapes=show_output_shapes,
    )

    # ------------------------------------------------------------------
    # Resolve core settings.
    # ------------------------------------------------------------------
    has_validation: bool = validation_bundle is not None
    run_qc: bool = bool(qc_config.get("run_qc", True))
    do_save: bool = bool(save_config.get("save", False))
    preproc_key: str = preprocessing_config["preproc_key"]

    if do_save and save_config.get("output_dir", None) is None:
        raise ValueError(
            "save_kwargs['output_dir'] must be provided when save_kwargs['save']=True."
        )

    if progress_enabled:
        print("Preprocessing train/validation bundles")
        print("------------------------------------")

    # ------------------------------------------------------------------
    # Small internal helper: print QC section headers.
    # ------------------------------------------------------------------
    def _print_qc_section(
        *,
        bundle_name: str,
        stage: str,
    ) -> None:
        """
        Print a visible section divider before QC plots.
        """
        if not progress_enabled:
            return

        if not qc_config.get("show_qc_sections", True):
            return

        display_name = bundle_name.replace("_bundle", "").replace("_", " ").title()
        stage_name = stage.upper()

        print("")
        print("=" * 72)
        print(f"{display_name} bundle - {stage_name} feature QC")
        print("=" * 72)

    # ------------------------------------------------------------------
    # Helper: remove columns that no longer exist from config lists.
    # ------------------------------------------------------------------
    def _filter_encoder_config_to_existing_columns(
        cfg: Dict[str, Any],
        existing_cols: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        Remove missing columns from encoder list settings after raw cleaning
        or high-cardinality handling.
        """
        cfg_out = dict(cfg)
        existing_set = set(existing_cols)

        list_keys = [
            "cat_cols",
            "ord_cols",
            "categorical_passthrough_cols",
            "ordinal_passthrough_cols",
        ]

        removed: Dict[str, List[str]] = {}

        for key in list_keys:
            original_cols = list(cfg_out.get(key) or [])
            retained_cols = [
                col for col in original_cols
                if col in existing_set
            ]
            removed_cols = [
                col for col in original_cols
                if col not in existing_set
            ]

            cfg_out[key] = retained_cols
            removed[key] = removed_cols

        if cfg_out.get("drop_cols") is not None:
            original_drop_cols = list(cfg_out.get("drop_cols") or [])
            retained_drop_cols = [
                col for col in original_drop_cols
                if col in existing_set
            ]
            removed_drop_cols = [
                col for col in original_drop_cols
                if col not in existing_set
            ]

            cfg_out["drop_cols"] = retained_drop_cols
            removed["drop_cols"] = removed_drop_cols
        else:
            removed["drop_cols"] = []

        return cfg_out, removed

    # ------------------------------------------------------------------
    # Step 1: Validate raw input bundles.
    # ------------------------------------------------------------------
    step_name = "Validate raw bundles"

    try:
        _start_step(step_name)

        for bundle_name, bundle in [("train_bundle", train_bundle)]:
            if "X_raw" not in bundle:
                raise KeyError(f"{bundle_name} is missing required key 'X_raw'.")

            if "y" not in bundle:
                raise KeyError(f"{bundle_name} is missing required key 'y'.")

            if "feature_names" not in bundle:
                raise KeyError(f"{bundle_name} is missing required key 'feature_names'.")

            if not isinstance(bundle["X_raw"], pd.DataFrame):
                raise TypeError(
                    f"{bundle_name}['X_raw'] must be a pandas DataFrame at this "
                    "stage. This function expects raw bundles created by the new "
                    "prepare_train_validation_bundles(...)."
                )

        if has_validation:
            if "X_raw" not in validation_bundle:
                raise KeyError("validation_bundle is missing required key 'X_raw'.")

            if "y" not in validation_bundle:
                raise KeyError("validation_bundle is missing required key 'y'.")

            if "feature_names" not in validation_bundle:
                raise KeyError(
                    "validation_bundle is missing required key 'feature_names'."
                )

            if not isinstance(validation_bundle["X_raw"], pd.DataFrame):
                raise TypeError(
                    "validation_bundle['X_raw'] must be a pandas DataFrame at this "
                    "stage. This function expects raw bundles created by the new "
                    "prepare_train_validation_bundles(...)."
                )

        validate_detail = {
            "train_X_raw_shape": tuple(train_bundle["X_raw"].shape),
            "validation_X_raw_shape": (
                tuple(validation_bundle["X_raw"].shape)
                if has_validation
                else None
            ),
            "has_validation": has_validation,
        }

        _ok_step(step_name, validate_detail)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 2: Resolve raw DataFrames.
    # ------------------------------------------------------------------
    step_name = "Resolve raw feature dataframes"

    try:
        _start_step(step_name)

        X_train_raw_df = train_bundle["X_raw"].copy()
        y_train = np.asarray(train_bundle["y"])

        train_raw_feature_names = list(train_bundle["feature_names"])
        X_train_raw_df.columns = train_raw_feature_names

        if has_validation:
            X_validation_raw_df = validation_bundle["X_raw"].copy()
            y_validation = np.asarray(validation_bundle["y"])

            validation_raw_feature_names = list(validation_bundle["feature_names"])
            X_validation_raw_df.columns = validation_raw_feature_names

            if list(X_validation_raw_df.columns) != list(X_train_raw_df.columns):
                raise ValueError(
                    "Train and validation raw feature columns do not match before "
                    "preprocessing. Run prepare_train_validation_bundles(...) first "
                    "or check validation feature alignment."
                )
        else:
            X_validation_raw_df = None
            y_validation = None

        _ok_step(step_name, X_train_raw_df)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 3: Clean raw training features and apply same columns to validation.
    # ------------------------------------------------------------------
    step_name = "Clean raw feature columns"

    try:
        _start_step(step_name)

        near_constant_feature_types = tuple(
            str(feature_type).lower()
            for feature_type in raw_feature_cleaning_config.get(
                "near_constant_feature_types",
                ("categorical", "ordinal"),
            )
        )

        near_constant_check_cols: Dict[str, str] = {}

        if "categorical" in near_constant_feature_types:
            for col in encoder_config.get("cat_cols") or []:
                near_constant_check_cols[str(col)] = "categorical"

            for col in encoder_config.get("categorical_passthrough_cols") or []:
                near_constant_check_cols[str(col)] = "categorical"

        if "ordinal" in near_constant_feature_types:
            for col in encoder_config.get("ord_cols") or []:
                near_constant_check_cols[str(col)] = "ordinal"

            for col in encoder_config.get("ordinal_passthrough_cols") or []:
                near_constant_check_cols[str(col)] = "ordinal"

        raw_feature_cleaning_call_config = dict(raw_feature_cleaning_config)
        raw_feature_cleaning_call_config["near_constant_check_cols"] = (
            near_constant_check_cols
        )

        # Clean the raw training feature matrix only.
        #
        # This is train-fitted raw feature cleaning:
        #   - duplicate-name cleanup is learned/applied on train,
        #   - high-missingness filtering is based on train missingness,
        #   - constant and near-constant filtering are based on train only.
        #
        # The retained feature_names_clean list becomes the raw-feature contract
        # that validation must follow.
        X_train_clean_values, feature_names_clean, raw_feature_cleaning_meta = (
            clean_raw_feature_columns(
                X=X_train_raw_df,
                feature_names=train_raw_feature_names,
                return_metadata=True,
                **raw_feature_cleaning_call_config,
            )
        )

        X_train_clean_df = pd.DataFrame(
            X_train_clean_values,
            columns=feature_names_clean,
        )

        if has_validation:
            # Validation must follow the same cleaned raw-feature contract as train.
            #
            # Important:
            # If the raw validation DataFrame still contains duplicate column names,
            # selecting columns by name can return duplicate columns. In pandas,
            # data[col] returns a DataFrame instead of a Series when col is duplicated,
            # which later breaks encoding.
            #
            # Therefore, if duplicate-name cleanup is enabled, remove duplicate
            # validation columns using the same keep-first rule before selecting
            # the train-retained feature_names_clean.
            X_validation_for_alignment = X_validation_raw_df.copy()

            if raw_feature_cleaning_config.get("drop_duplicate_names", True):
                validation_duplicate_mask = (
                    X_validation_for_alignment.columns.duplicated(keep="first")
                )

                validation_duplicate_names = list(
                    X_validation_for_alignment.columns[validation_duplicate_mask]
                )

                if validation_duplicate_names:
                    X_validation_for_alignment = X_validation_for_alignment.loc[
                        :,
                        ~validation_duplicate_mask,
                    ].copy()

                    raw_feature_cleaning_meta[
                        "validation_duplicate_name_columns_removed"
                    ] = validation_duplicate_names

                    raw_feature_cleaning_meta[
                        "n_validation_duplicate_name_columns_removed"
                    ] = len(validation_duplicate_names)
                else:
                    raw_feature_cleaning_meta[
                        "validation_duplicate_name_columns_removed"
                    ] = []

                    raw_feature_cleaning_meta[
                        "n_validation_duplicate_name_columns_removed"
                    ] = 0
            else:
                raw_feature_cleaning_meta[
                    "validation_duplicate_name_columns_removed"
                ] = []

                raw_feature_cleaning_meta[
                    "n_validation_duplicate_name_columns_removed"
                ] = 0

            missing_validation_cols = [
                col for col in feature_names_clean
                if col not in X_validation_for_alignment.columns
            ]

            if missing_validation_cols:
                raise KeyError(
                    "Validation bundle is missing columns retained after train "
                    f"raw-feature cleaning: {missing_validation_cols}"
                )

            # Keep exactly the train-retained raw features and preserve train order.
            X_validation_clean_df = X_validation_for_alignment.loc[
                :,
                feature_names_clean,
            ].copy()

            # Final defensive check.
            # After validation alignment, column names must be unique.
            validation_duplicate_after_alignment = list(
                X_validation_clean_df.columns[
                    X_validation_clean_df.columns.duplicated(keep=False)
                ]
            )

            if validation_duplicate_after_alignment:
                raise ValueError(
                    "Validation raw-feature alignment still contains duplicate "
                    "columns after applying the cleaned train feature contract: "
                    f"{validation_duplicate_after_alignment}"
                )

        else:
            X_validation_clean_df = None

            raw_feature_cleaning_meta[
                "validation_duplicate_name_columns_removed"
            ] = []

            raw_feature_cleaning_meta[
                "n_validation_duplicate_name_columns_removed"
            ] = 0

        encoder_config, encoder_removed_after_raw_cleaning = (
            _filter_encoder_config_to_existing_columns(
                encoder_config,
                feature_names_clean,
            )
        )

        raw_feature_cleaning_meta[
            "encoder_config_removed_after_raw_cleaning"
        ] = encoder_removed_after_raw_cleaning

        high_missing_dropped_names = [
            item["feature_name"]
            for item in raw_feature_cleaning_meta.get(
                "high_missing_columns_dropped",
                [],
            )
        ]

        if high_missing_dropped_names:
            max_preview_cols = 8
            high_missing_preview_names = high_missing_dropped_names[:max_preview_cols]
            high_missing_preview = ", ".join(high_missing_preview_names)

            if len(high_missing_dropped_names) > max_preview_cols:
                high_missing_preview += (
                    f", ... +{len(high_missing_dropped_names) - max_preview_cols} more"
                )

            high_missing_detail = (
                f"{raw_feature_cleaning_meta['n_high_missing_columns_dropped']} "
                f"({high_missing_preview})"
            )
        else:
            high_missing_detail = "0"

        raw_cleaning_detail = (
            f"raw features "
            f"{raw_feature_cleaning_meta['n_features_before']} -> "
            f"{raw_feature_cleaning_meta['n_features_after']}; "
            f"duplicate-name dropped="
            f"{raw_feature_cleaning_meta['n_duplicate_name_columns_dropped']}; "
            f"validation duplicate-name aligned="
            f"{raw_feature_cleaning_meta['n_validation_duplicate_name_columns_removed']}; "
            f"high-missing dropped="
            f"{high_missing_detail}; "
            f"constant dropped="
            f"{raw_feature_cleaning_meta['n_constant_columns_dropped']}; "
            f"near-constant dropped="
            f"{raw_feature_cleaning_meta['n_near_constant_features_dropped']}"
        )

        _ok_step(step_name, raw_cleaning_detail)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 4: Apply high-cardinality handling.
    # ------------------------------------------------------------------
    step_name = "Handle high-cardinality features"

    if high_cardinality_config.get("enabled", False):
        try:
            _start_step(step_name)

            X_train_feature_df, X_validation_feature_df, high_cardinality_meta = (
                apply_high_cardinality_handling_train_validation(
                    X_train_clean_df,
                    X_validation_clean_df,
                    high_cardinality_kwargs=high_cardinality_config,
                )
            )

            multi_label_indicator_cols: List[str] = []

            for _, col_meta in high_cardinality_meta.get("multi_label", {}).items():
                multi_label_indicator_cols.extend(
                    list(col_meta.get("new_indicator_cols", []))
                )

            if multi_label_indicator_cols:
                existing_cat_passthrough = list(
                    encoder_config.get("categorical_passthrough_cols") or []
                )

                encoder_config["categorical_passthrough_cols"] = list(
                    dict.fromkeys(
                        existing_cat_passthrough + multi_label_indicator_cols
                    )
                )

            encoder_config, encoder_removed_after_high_cardinality = (
                _filter_encoder_config_to_existing_columns(
                    encoder_config,
                    list(X_train_feature_df.columns),
                )
            )

            high_cardinality_meta[
                "encoder_config_removed_after_high_cardinality"
            ] = encoder_removed_after_high_cardinality

            high_cardinality_detail = (
                f"train features "
                f"{high_cardinality_meta['n_train_features_before']} -> "
                f"{high_cardinality_meta['n_train_features_after']}"
            )

            if has_validation:
                high_cardinality_detail += (
                    f"; validation features "
                    f"{high_cardinality_meta['n_validation_features_before']} -> "
                    f"{high_cardinality_meta['n_validation_features_after']}"
                )

            _ok_step(step_name, high_cardinality_detail)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    else:
        X_train_feature_df = X_train_clean_df.copy()
        X_validation_feature_df = (
            X_validation_clean_df.copy()
            if X_validation_clean_df is not None
            else None
        )

        high_cardinality_meta = {
            "enabled": False,
            "n_train_features_before": int(X_train_clean_df.shape[1]),
            "n_train_features_after": int(X_train_feature_df.shape[1]),
            "n_validation_features_before": (
                int(X_validation_clean_df.shape[1])
                if X_validation_clean_df is not None
                else None
            ),
            "n_validation_features_after": (
                int(X_validation_feature_df.shape[1])
                if X_validation_feature_df is not None
                else None
            ),
        }

        _skip_step(
            step_name,
            "high_cardinality_kwargs['enabled'] is False",
        )

    # ------------------------------------------------------------------
    # Step 5: Impute raw categorical/ordinal features before encoding.
    # ------------------------------------------------------------------
    step_name = "Impute raw categorical/ordinal features"

    if raw_cat_ord_impute_config.get("enabled", True):
        try:
            _start_step(step_name)

            X_train_feature_df, X_validation_feature_df, raw_cat_ord_impute_meta = (
                impute_raw_categorical_ordinal_train_validation(
                    X_train_feature_df,
                    X_validation_feature_df,
                    encoder_config=encoder_config,
                    raw_categorical_ordinal_impute_kwargs=raw_cat_ord_impute_config,
                )
            )

            raw_cat_ord_detail = (
                f"columns considered="
                f"{raw_cat_ord_impute_meta['n_columns_considered']}; "
                f"columns imputed="
                f"{raw_cat_ord_impute_meta['n_columns_imputed']}"
            )

            _ok_step(step_name, raw_cat_ord_detail)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    else:
        raw_cat_ord_impute_meta = {
            "enabled": False,
            "columns": {},
            "n_columns_considered": 0,
            "n_columns_imputed": 0,
        }

        _skip_step(
            step_name,
            "raw_categorical_ordinal_impute_kwargs['enabled'] is False",
        )

    # ------------------------------------------------------------------
    # Step 6: Encode training features.
    # ------------------------------------------------------------------
    step_name = "Encode train features"

    try:
        _start_step(step_name)

        X_train_encoded, train_encoding_meta = encode_categorical_and_ordinal(
            X_train_feature_df,
            return_metadata=True,
            **encoder_config,
        )

        _ok_step(step_name, X_train_encoded)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 7: Encode validation features like training.
    # ------------------------------------------------------------------
    step_name = "Encode validation features"

    if has_validation:
        try:
            _start_step(step_name)

            X_validation_encoded, validation_encoding_meta = (
                encode_categorical_and_ordinal_like_train(
                    X_validation_feature_df,
                    train_encoding_meta=train_encoding_meta,
                    encoder_kwargs=encoder_config,
                    strict_categories=encoder_config.get("strict_categories", True),
                    return_metadata=True,
                )
            )

            _ok_step(step_name, X_validation_encoded)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    else:
        X_validation_encoded = None
        validation_encoding_meta = None

        _skip_step(step_name, "validation_bundle is None")

    # ------------------------------------------------------------------
    # Step 8: Sanitize encoded feature names.
    # ------------------------------------------------------------------
    step_name = "Sanitize encoded feature names"

    if sanitize_config.get("enabled", True):
        try:
            _start_step(step_name)

            sanitize_call_kwargs = dict(sanitize_config)
            sanitize_call_kwargs.pop("enabled", None)

            original_encoded_feature_names = list(X_train_encoded.columns)

            encoded_feature_names, feature_name_sanitization_meta = (
                sanitize_feature_names(
                    original_encoded_feature_names,
                    return_metadata=True,
                    **sanitize_call_kwargs,
                )
            )

            feature_name_mapping = feature_name_sanitization_meta[
                "original_to_sanitized"
            ]

            X_train_encoded = X_train_encoded.copy()
            X_train_encoded.columns = encoded_feature_names

            if X_validation_encoded is not None:
                X_validation_encoded = X_validation_encoded.copy()
                X_validation_encoded.columns = encoded_feature_names

            train_encoding_meta = update_encoding_metadata_feature_names(
                train_encoding_meta,
                feature_name_mapping=feature_name_mapping,
            )

            validation_encoding_meta = update_encoding_metadata_feature_names(
                validation_encoding_meta,
                feature_name_mapping=feature_name_mapping,
            )

            _ok_step(step_name, encoded_feature_names)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    else:
        encoded_feature_names = list(X_train_encoded.columns)
        feature_name_sanitization_meta = None

        _skip_step(
            step_name,
            "sanitize_feature_names_kwargs['enabled'] is False",
        )

    # ------------------------------------------------------------------
    # Step 9: Build encoded bundles.
    # ------------------------------------------------------------------
    step_name = "Build encoded bundles"

    try:
        _start_step(step_name)

        train_bundle_preproc = dict(train_bundle)

        train_bundle_preproc.update(
            {
                "X_raw": X_train_encoded.to_numpy(dtype=np.float32, copy=False),
                "y": y_train,
                "feature_names": encoded_feature_names,
                "feature_name_to_idx": {
                    name: i for i, name in enumerate(encoded_feature_names)
                },
                "feature_encoding_metadata": train_encoding_meta,
                "validation_feature_encoding_metadata": None,
                "feature_name_sanitization": feature_name_sanitization_meta,
                "raw_feature_cleaning": raw_feature_cleaning_meta,
                "high_cardinality_handling": high_cardinality_meta,
                "raw_categorical_ordinal_imputation": raw_cat_ord_impute_meta,
                "is_raw_split": False,
                "is_encoded": True,
                "is_preprocessed": False,
            }
        )

        if has_validation:
            validation_bundle_preproc = dict(validation_bundle)

            validation_bundle_preproc.update(
                {
                    "X_raw": X_validation_encoded.to_numpy(
                        dtype=np.float32,
                        copy=False,
                    ),
                    "y": y_validation,
                    "feature_names": encoded_feature_names,
                    "feature_name_to_idx": {
                        name: i for i, name in enumerate(encoded_feature_names)
                    },
                    "feature_encoding_metadata": train_encoding_meta,
                    "validation_feature_encoding_metadata": validation_encoding_meta,
                    "feature_name_sanitization": feature_name_sanitization_meta,
                    "raw_feature_cleaning": raw_feature_cleaning_meta,
                    "high_cardinality_handling": high_cardinality_meta,
                    "raw_categorical_ordinal_imputation": raw_cat_ord_impute_meta,
                    "is_raw_split": False,
                    "is_encoded": True,
                    "is_preprocessed": False,
                }
            )
        else:
            validation_bundle_preproc = None

        encoded_bundle_detail = {
            "train_encoded_shape": tuple(train_bundle_preproc["X_raw"].shape),
            "validation_encoded_shape": (
                tuple(validation_bundle_preproc["X_raw"].shape)
                if validation_bundle_preproc is not None
                else None
            ),
            "n_encoded_features": len(encoded_feature_names),
        }

        _ok_step(step_name, encoded_bundle_detail)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 10: Resolve metadata for data_preprocessing_pipeline.
    # ------------------------------------------------------------------
    step_name = "Resolve preprocessing metadata"

    try:
        _start_step(step_name)

        meta = preprocessing_config.get("meta", "auto")

        if meta == "auto":
            meta = train_bundle_preproc.get("feature_encoding_metadata", None)

        preprocessing_meta_detail = {
            "meta_source": "feature_encoding_metadata" if meta is not None else None,
            "preproc_key": preproc_key,
        }

        _ok_step(step_name, preprocessing_meta_detail)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 11: Fit preprocessing on training only.
    # ------------------------------------------------------------------
    step_name = "Fit train preprocessing"

    try:
        _start_step(step_name)

        train_bundle_preproc = data_preprocessing_pipeline(
            bundle=train_bundle_preproc,
            lower_q=preprocessing_config["lower_q"],
            upper_q=preprocessing_config["upper_q"],
            continuous_impute_strategy=preprocessing_config[
                "continuous_impute_strategy"
            ],
            categorical_impute_strategy=preprocessing_config[
                "categorical_impute_strategy"
            ],
            ordinal_impute_strategy=preprocessing_config[
                "ordinal_impute_strategy"
            ],
            preproc_key=preproc_key,
            meta=meta,
        )

        train_bundle_preproc["is_preprocessed"] = True

        _ok_step(step_name, train_bundle_preproc)

    except Exception as err:
        _fail_step(step_name, err)
        raise

    # ------------------------------------------------------------------
    # Step 12: Apply fitted preprocessing to validation.
    # ------------------------------------------------------------------
    step_name = "Transform validation"

    if has_validation:
        try:
            _start_step(step_name)

            validation_bundle_preproc = preprocessing_transfer_pipeline(
                bundle_orig=train_bundle_preproc,
                bundle=validation_bundle_preproc,
                preproc_key=preproc_key,
                require_scaler=transfer_config["require_scaler"],
            )

            validation_bundle_preproc["is_preprocessed"] = True

            _ok_step(step_name, validation_bundle_preproc)

        except Exception as err:
            _fail_step(step_name, err)
            raise

    else:
        _skip_step(step_name, "validation_bundle is None")

    # ------------------------------------------------------------------
    # Step 13: Initialize QC containers.
    # ------------------------------------------------------------------
    train_summaries: Dict[str, pd.DataFrame] = {}
    validation_summaries: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Helper: run QC for one bundle and one stage.
    # ------------------------------------------------------------------
    def _run_qc_for_bundle_stage(
        *,
        bundle: Dict[str, Any],
        bundle_name: str,
        stage: str,
    ) -> pd.DataFrame:
        """
        Run missingness plot, summary table, and raincloud plot for one bundle
        at one stage: raw or scaled.
        """

        matrix_key = "X_raw" if stage == "raw" else "X_scaled"

        if matrix_key not in bundle:
            raise KeyError(
                f"Cannot run {stage!r} QC for {bundle_name}: "
                f"bundle is missing key {matrix_key!r}."
            )

        _print_qc_section(
            bundle_name=bundle_name,
            stage=stage,
        )

        visualize_missingness(
            X_raw=bundle[matrix_key],
            feature_names=bundle["feature_names"],
            kind=qc_config["missingness_kind"],
            max_features=qc_config["max_features"],
            figsize=qc_config["missingness_figsize"],
            fontsize=qc_config["missingness_fontsize"],
            color=qc_config["missingness_color"],
            sort=qc_config["missingness_sort"],
        )

        summary_df = summarize_feature_matrix(
            bundle[matrix_key],
            bundle["feature_names"],
        )

        plot_feature_stat_raincloud_by_type(
            summary_df=summary_df,
            feature_encoding_metadata=bundle.get("feature_encoding_metadata", None),
            stat=qc_config["summary_stat"],
            figsize=qc_config["raincloud_figsize"],
            font_size=qc_config["raincloud_font_size"],
            show_points=True,
            violin_half=qc_config["violin_half"],
        )


        return summary_df

    # ------------------------------------------------------------------
    # Step 14: Run train QC.
    # ------------------------------------------------------------------
    if run_qc:
        for stage in qc_stages:
            step_name = f"Run train {stage} QC"

            try:
                _start_step(step_name)

                train_summaries[stage] = _run_qc_for_bundle_stage(
                    bundle=train_bundle_preproc,
                    bundle_name="train_bundle",
                    stage=stage,
                )

                _ok_step(step_name, train_summaries[stage])

            except Exception as err:
                _fail_step(step_name, err)
                raise

    else:
        _skip_step("Run train QC", "qc_kwargs['run_qc'] is False")

    # ------------------------------------------------------------------
    # Step 15: Run validation QC.
    # ------------------------------------------------------------------
    if run_qc and has_validation:
        for stage in qc_stages:
            step_name = f"Run validation {stage} QC"

            try:
                _start_step(step_name)

                validation_summaries[stage] = _run_qc_for_bundle_stage(
                    bundle=validation_bundle_preproc,
                    bundle_name="validation_bundle",
                    stage=stage,
                )

                _ok_step(step_name, validation_summaries[stage])

            except Exception as err:
                _fail_step(step_name, err)
                raise

    elif run_qc and not has_validation:
        _skip_step("Run validation QC", "validation_bundle is None")

    # ------------------------------------------------------------------
    # Step 16: Build metadata output.
    # ------------------------------------------------------------------
    preproc_meta: Dict[str, Any] = {
        "has_validation": has_validation,
        "preproc_key": preproc_key,

        "raw_feature_cleaning_kwargs": raw_feature_cleaning_config,
        "high_cardinality_kwargs": high_cardinality_config,
        "raw_categorical_ordinal_impute_kwargs": raw_cat_ord_impute_config,
        "encoder_kwargs": encoder_config,
        "sanitize_feature_names_kwargs": sanitize_config,
        "preprocessing_kwargs": preprocessing_config,
        "transfer_kwargs": transfer_config,
        "qc_kwargs": qc_config,
        "save_kwargs": save_config,
        "progress_kwargs": progress_config,

        "raw_feature_cleaning": raw_feature_cleaning_meta,
        "high_cardinality_handling": high_cardinality_meta,
        "raw_categorical_ordinal_imputation": raw_cat_ord_impute_meta,
        "feature_encoding_metadata": train_encoding_meta,
        "validation_feature_encoding_metadata": validation_encoding_meta,
        "feature_name_sanitization": feature_name_sanitization_meta,

        "train_shape_raw_input": tuple(X_train_raw_df.shape),
        "train_shape_after_raw_cleaning": tuple(X_train_clean_df.shape),
        "train_shape_after_high_cardinality": tuple(X_train_feature_df.shape),
        "train_shape_encoded": tuple(train_bundle_preproc["X_raw"].shape),
        "train_shape_scaled": tuple(train_bundle_preproc["X_scaled"].shape),

        "validation_shape_raw_input": (
            tuple(X_validation_raw_df.shape)
            if X_validation_raw_df is not None
            else None
        ),
        "validation_shape_after_raw_cleaning": (
            tuple(X_validation_clean_df.shape)
            if X_validation_clean_df is not None
            else None
        ),
        "validation_shape_after_high_cardinality": (
            tuple(X_validation_feature_df.shape)
            if X_validation_feature_df is not None
            else None
        ),
        "validation_shape_encoded": (
            tuple(validation_bundle_preproc["X_raw"].shape)
            if validation_bundle_preproc is not None
            else None
        ),
        "validation_shape_scaled": (
            tuple(validation_bundle_preproc["X_scaled"].shape)
            if validation_bundle_preproc is not None
            and "X_scaled" in validation_bundle_preproc
            else None
        ),

        "feature_names": encoded_feature_names,
        "n_features": len(encoded_feature_names),

        "train_summaries": train_summaries,
        "validation_summaries": validation_summaries,
        "train_summary_raw": train_summaries.get("raw"),
        "train_summary_scaled": train_summaries.get("scaled"),
        "validation_summary_raw": validation_summaries.get("raw"),
        "validation_summary_scaled": validation_summaries.get("scaled"),

        "saved_paths": {},
    }

    if return_dataframes:
        preproc_meta["X_train_raw_df"] = X_train_raw_df
        preproc_meta["X_validation_raw_df"] = X_validation_raw_df
        preproc_meta["X_train_clean_df"] = X_train_clean_df
        preproc_meta["X_validation_clean_df"] = X_validation_clean_df
        preproc_meta["X_train_feature_df"] = X_train_feature_df
        preproc_meta["X_validation_feature_df"] = X_validation_feature_df
        preproc_meta["X_train_encoded"] = X_train_encoded
        preproc_meta["X_validation_encoded"] = X_validation_encoded

    if return_progress_log:
        preproc_meta["progress_log"] = progress_log

    # ------------------------------------------------------------------
    # Step 17: Optionally save outputs.
    # ------------------------------------------------------------------
    if do_save:
        step_name = "Save outputs"

        try:
            _start_step(step_name)

            output_dir = save_config["output_dir"]
            train_prefix = save_config["train_prefix"]
            validation_prefix = save_config["validation_prefix"]
            meta_prefix = save_config["meta_prefix"]
            compress = save_config["compress"]
            save_metadata = save_config["save_metadata"]

            save_sidecar_metadata: Dict[str, Any] = {
                "has_validation": has_validation,
                "preproc_key": preproc_key,
                "train_shape_encoded": preproc_meta["train_shape_encoded"],
                "train_shape_scaled": preproc_meta["train_shape_scaled"],
                "validation_shape_encoded": preproc_meta[
                    "validation_shape_encoded"
                ],
                "validation_shape_scaled": preproc_meta[
                    "validation_shape_scaled"
                ],
                "n_features": preproc_meta["n_features"],
            }

            train_out = ut.save_all_results(
                output_dir=output_dir,
                all_results=train_bundle_preproc,
                prefix=train_prefix,
                compress=compress,
                metadata=save_sidecar_metadata if save_metadata else None,
            )

            preproc_meta["saved_paths"]["train_bundle_dir"] = str(train_out)

            if validation_bundle_preproc is not None:
                validation_out = ut.save_all_results(
                    output_dir=output_dir,
                    all_results=validation_bundle_preproc,
                    prefix=validation_prefix,
                    compress=compress,
                    metadata=save_sidecar_metadata if save_metadata else None,
                )

                preproc_meta["saved_paths"]["validation_bundle_dir"] = str(
                    validation_out
                )

            meta_out = ut.save_all_results(
                output_dir=output_dir,
                all_results=preproc_meta,
                prefix=meta_prefix,
                compress=compress,
                metadata=save_sidecar_metadata if save_metadata else None,
            )

            preproc_meta["saved_paths"]["preproc_meta_dir"] = str(meta_out)

            _ok_step(step_name, preproc_meta["saved_paths"])

        except Exception as err:
            _fail_step(step_name, err)
            raise

    else:
        _skip_step("Save outputs", "save_kwargs['save'] is False")

    # ------------------------------------------------------------------
    # Finalize.
    # ------------------------------------------------------------------
    if progress_enabled:
        print("------------------------------------")
        print("[OK] Pipeline complete")

    return train_bundle_preproc, validation_bundle_preproc, preproc_meta



# ---------------------------------------------------------------------
# Explore data missingness and run data preprocessing
# ---------------------------------------------------------------------


def summarize_feature_matrix(
    X_raw: np.ndarray,
    feature_names: List[str],
    percentiles: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for the feature matrix using pandas.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Feature matrix (e.g., output from stack_features_with_groups).
    feature_names : list[str]
        Names of the features, ordered to match the columns of X_raw.
    percentiles : list[float], optional
        List of percentiles to include in the output (values between 0 and 1).
        If None, uses pandas' default: [0.25, 0.5, 0.75].

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame where rows are summary statistics (count, mean, std, min,
        selected percentiles, max) and columns are feature names.
    """
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]

    df = pd.DataFrame(X_raw, columns=feature_names)
    summary_df = df.describe(percentiles=percentiles)

    return summary_df


def plot_feature_stat_distribution(
    summary_df: pd.DataFrame,
    stat: str = "std",
    kind: Literal["hist", "box"] = "hist",
    bins: int = 50,
    figsize=(8, 3),
    font_size=12,
    xlabel: str | None = None,
) -> None:
    if stat not in summary_df.index:
        raise ValueError(
            f"stat='{stat}' not found in summary_df.index. "
            f"Available: {list(summary_df.index)}"
        )

    values = summary_df.loc[stat].values.astype(float)

    # Auto-generate a more informative x-label
    if xlabel is None:
        if stat.endswith("%") and stat[:-1].isdigit():
            p = stat
            extra = " (median)" if p == "50%" else ""
            xlabel = f"Per-feature {p} percentile{extra} (across samples)"
        else:
            stat_word = "standard deviation" if stat == "std" else stat
            xlabel = f"Per-feature {stat_word} (across samples)"

    plt.figure(figsize=figsize)

    if kind == "hist":
        plt.hist(values, bins=bins)
        plt.xlabel(xlabel, fontsize=font_size, fontweight="bold")
        plt.ylabel("Number of EEG features", fontsize=font_size, fontweight="bold")
        plt.title(
            f"Distribution across EEG features: {stat}",
            fontsize=font_size,
            fontweight="bold",
        )

    elif kind == "box":
        plt.boxplot(values, vert=False)
        plt.xlabel(xlabel, fontsize=font_size, fontweight="bold")
        plt.title(
            f"Boxplot across EEG features: {stat}",
            fontsize=font_size,
            fontweight="bold",
        )

    else:
        raise ValueError("kind must be 'hist' or 'box'.")

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.show()



def plot_feature_stat_raincloud(
    summary_df: pd.DataFrame,
    stat: str = "std",
    title: Optional[str] = None,
    feature_label: str = "features",
    show_points: bool = True,
    figsize: tuple[float, float] = (8, 3),
    font_size: int = 12,
    xlabel: Optional[str] = None,
    base_color: str = "#FFB400",
    violin_color: Optional[str] = None,
    point_color: Optional[str] = None,
    jitter_width: float = 0.05,
    point_size: float = 10,
    point_alpha: float = 0.15,
    point_edgecolors: str = "none",
    box_linewidth: float = 1.5,
    median_linewidth: float = 2.0,
    violin_alpha: float = 0.5,
    violin_edgecolor: str = "black",
    violin_linewidth: float = 1.0,
    violin_half: Literal["full", "left", "right"] = "left",
) -> None:
    """
    Raincloud-style plot for one feature statistic across a set of features.

    Example:
        stat="mean" plots one mean value per feature.
        stat="std" plots one standard deviation value per feature.

    summary_df:
        Rows are statistics, columns are feature names.
    """

    if stat not in summary_df.index:
        raise ValueError(
            f"stat='{stat}' not found in summary_df.index. "
            f"Available: {list(summary_df.index)}"
        )

    values: np.ndarray = summary_df.loc[stat].values.astype(float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        print(f"[SKIP] No finite values for {feature_label}: stat={stat}")
        return

    if xlabel is None:
        if stat.endswith("%") and stat[:-1].isdigit():
            p = stat
            extra = " median" if p == "50%" else ""
            xlabel = f"Per-feature {p} percentile{extra} across samples"
        else:
            stat_word = "standard deviation" if stat == "std" else stat
            xlabel = f"Per-feature {stat_word} across samples"

    if title is None:
        title = f"{feature_label.title()}: per-feature {stat}"

    if violin_color is None:
        violin_color = base_color

    if point_color is None:
        point_color = base_color

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    pos: float = 0.0
    viol_offset: float = -0.20
    box_offset: float = 0.20

    violin_center_x: float = pos + viol_offset
    box_center_x: float = pos + box_offset

    viol_parts = ax.violinplot(
        [values],
        positions=[violin_center_x],
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in viol_parts["bodies"]:
        body.set_facecolor(violin_color)
        body.set_edgecolor(violin_edgecolor)
        body.set_alpha(violin_alpha)
        body.set_linewidth(violin_linewidth)

        if violin_half != "full":
            path = body.get_paths()[0]
            verts = path.vertices

            if violin_half == "right":
                verts[:, 0] = np.minimum(verts[:, 0], violin_center_x)
            elif violin_half == "left":
                verts[:, 0] = np.maximum(verts[:, 0], violin_center_x)

            path.vertices = verts

    box_parts = ax.boxplot(
        [values],
        positions=[box_center_x],
        widths=0.25,
        patch_artist=True,
        showfliers=False,
    )

    for box, median in zip(box_parts["boxes"], box_parts["medians"]):
        box.set_facecolor("none")
        box.set_edgecolor("black")
        box.set_linewidth(box_linewidth)
        median.set_color("black")
        median.set_linewidth(median_linewidth)
        box.set_zorder(3)

    if show_points:
        x_jitter: np.ndarray = box_center_x + np.random.uniform(
            -jitter_width,
            jitter_width,
            size=len(values),
        )

        ax.scatter(
            x_jitter,
            values,
            s=point_size,
            alpha=point_alpha,
            color=point_color,
            edgecolors=point_edgecolors,
            zorder=1,
        )

    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)

    ax.set_title(title, fontsize=font_size + 2, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight="bold")
    ax.set_ylabel("Statistic value", fontsize=font_size, fontweight="bold")

    ax.tick_params(axis="both", labelsize=font_size)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()
    plt.show()




    
def visualize_missingness(
    X_raw: np.ndarray,
    feature_names: List[str],
    kind: Literal["matrix", "bar"] = "matrix",
    max_features: Optional[int] = None,
    **msno_kwargs: Any,
) -> None:
    """
    Visualize missing data patterns in the feature matrix using `missingno`.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Feature matrix, potentially containing NaNs for missing values.
    feature_names : list[str]
        Names of the features, ordered to match the columns of X_raw.
    kind : {"matrix", "bar"}, default "matrix"
        Type of plot:
          - "matrix" : sparkline-style overview of missingness per sample/feature
          - "bar"    : bar plot showing count of non-missing per feature
    max_features : int, optional
        If set and the number of features is larger than this, only the first
        `max_features` columns are visualized (to keep plots readable).
    **msno_kwargs :
        Additional keyword arguments passed directly to the underlying
        `missingno.matrix` or `missingno.bar` call.
        Examples: figsize=(16, 6), fontsize=12, color="maroon", sort="ascending".
    """
    df = pd.DataFrame(X_raw, columns=feature_names)

    if max_features is not None and df.shape[1] > max_features:
        df = df.iloc[:, :max_features]

    # Normalize color if provided as a name; otherwise leave as-is
    if "color" in msno_kwargs and isinstance(msno_kwargs["color"], str):
        msno_kwargs = {**msno_kwargs}  # shallow copy so we don't mutate caller's dict
        msno_kwargs["color"] = mcolors.to_rgb(msno_kwargs["color"])

    if kind == "matrix":
        msno.matrix(df, **msno_kwargs)
    elif kind == "bar":
        msno.bar(df, **msno_kwargs)
    else:
        raise ValueError(f"Unknown kind='{kind}'. Use 'matrix' or 'bar'.")

    #plt.tight_layout()
    plt.show()


# Winsorization / outlier capping
def cap_outliers_percentile(
    X_raw: np.ndarray,
    feature_names: List[str],
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Winsorize features column-wise by capping values at given percentiles.

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix.
    feature_names : list[str]
        Names of the features, ordered to match the columns of X_raw.
    lower_q : float, default 0.05
        Lower percentile (between 0 and 1). Values below this will be
        set to the lower_q percentile value for that feature.
    upper_q : float, default 0.95
        Upper percentile (between 0 and 1). Values above this will be
        set to the upper_q percentile value for that feature.

    Returns
    -------
    X_capped : np.ndarray, shape (n_samples, n_features)
        Feature matrix after percentile capping.
    caps_df : pd.DataFrame
        DataFrame with index = feature_names and two columns:
        'lower' and 'upper', containing the percentile cutoffs used
        for each feature.
    """
    # Ensure a stable numeric dtype for quantiles and clipping
    X = np.asarray(X_raw, dtype=np.float32)

    # Compute per-feature percentiles (column-wise)
    # Note: use np.nanquantile if your matrix can contain NaNs.
    lower = np.nanquantile(X, lower_q, axis=0).astype(np.float32, copy=False)
    upper = np.nanquantile(X, upper_q, axis=0).astype(np.float32, copy=False)

    # Store caps in a small DataFrame (handy for inspection/debug)
    # Keep index aligned with feature_names, just like the original version.
    caps_df = pd.DataFrame({"lower": lower, "upper": upper}, index=feature_names)

    # Apply capping (winsorization)
    # np.clip supports per-column bounds when lower/upper are 1D arrays of length n_features
    X_capped = np.clip(X, lower, upper).astype(np.float32, copy=False)

    return X_capped, caps_df


# Data Standardization
def standardize_features(
    X_raw: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Apply column-wise standardization (zero mean, unit variance per feature).

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Raw feature matrix.

    Returns
    -------
    X_scaled : np.ndarray, shape (n_samples, n_features)
        Standardized feature matrix.
    scaler : StandardScaler
        Fitted scaler (so you can apply the same transform to new data).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    return X_scaled, scaler


# Missing Value Imputation
def impute_missing_features(
    X_raw: np.ndarray,
    strategy: str = "median",
) -> Tuple[np.ndarray, SimpleImputer]:
    """
    Impute missing values column-wise using a simple strategy
    (median by default).

    Parameters
    ----------
    X_raw : np.ndarray, shape (n_samples, n_features)
        Feature matrix with possible NaNs.
    strategy : {"mean", "median", "most_frequent", "constant"}, default "median"
        Imputation strategy passed to sklearn.SimpleImputer.

    Returns
    -------
    X_imputed : np.ndarray
        Feature matrix with NaNs filled in.
    imputer : SimpleImputer
        Fitted imputer (so you can apply the same transform to new data).
    """
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X_raw)
    return X_imputed, imputer


def impute_categorical_ordinal_mode(
    X_raw: np.ndarray,
    feature_names: List[str],
    meta: Dict[str, Any],
    *,
    categorical_impute_strategy: str = "mode",
    ordinal_impute_strategy: str = "mode",
    categorical_types: Tuple[str, ...] = (
        "onehot",
        "categorical_passthrough",
    ),
    ordinal_types: Tuple[str, ...] = (
        "ordinal",
        "ordinal_passthrough",
    ),
) -> Tuple[np.ndarray, SimpleImputer, List[int]]:
    """
    Impute categorical and ordinal columns identified by encoder metadata.

    Currently, categorical and ordinal features are mode-imputed using
    SimpleImputer(strategy="most_frequent").

    Parameters
    ----------
    X_raw:
        Feature matrix.

    feature_names:
        Feature names aligned with columns of X_raw.

    meta:
        Encoder metadata containing meta["output_to_source"].

    categorical_impute_strategy:
        Imputation strategy for categorical/discrete features.
        Currently only "mode" is supported.

    ordinal_impute_strategy:
        Imputation strategy for ordinal/discrete features.
        Currently only "mode" is supported.

    categorical_types:
        Metadata feature types treated as categorical for mode imputation.

    ordinal_types:
        Metadata feature types treated as ordinal for mode imputation.

    Returns
    -------
    X_imputed:
        Same shape as X_raw, with categorical/ordinal columns imputed.

    imputer:
        Fitted SimpleImputer(strategy="most_frequent") on categorical/ordinal columns.

    idx:
        Indices of columns that were imputed.
    """

    # Validate supported categorical imputation strategy.
    if categorical_impute_strategy != "mode":
        raise ValueError(
            "Only categorical_impute_strategy='mode' is currently supported."
        )

    # Validate supported ordinal imputation strategy.
    if ordinal_impute_strategy != "mode":
        raise ValueError(
            "Only ordinal_impute_strategy='mode' is currently supported."
        )

    # Convert input to float array for sklearn compatibility.
    X = np.asarray(X_raw, dtype=np.float32)

    # Extract output-to-source metadata.
    ots = meta.get("output_to_source", {})

    # Combined feature types that should use mode imputation.
    cat_ord_types = categorical_types + ordinal_types

    # Identify columns that should be mode-imputed.
    idx = [
        i for i, name in enumerate(feature_names)
        if (ots.get(name) or {}).get("type") in cat_ord_types
    ]

    # If there are no categorical/ordinal columns, return a copy unchanged.
    if not idx:
        return X.copy(), SimpleImputer(strategy="most_frequent"), []

    # Copy array before modifying.
    X_out = X.copy()

    # Fit mode imputer on categorical/ordinal columns only.
    imputer = SimpleImputer(strategy="most_frequent")

    # Apply mode imputation.
    X_out[:, idx] = imputer.fit_transform(X[:, idx]).astype(
        np.float32,
        copy=False,
    )

    return X_out, imputer, idx


def data_preprocessing_pipeline(
    bundle: Dict[str, Any],
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    continuous_impute_strategy: str = "median",
    categorical_impute_strategy: str = "mode",
    ordinal_impute_strategy: str = "mode",
    preproc_key: str = "preproc",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Preprocess a bundle's feature matrix in a fixed, reproducible order and
    attach the fitted preprocessing artifacts back onto the bundle.

    This function assumes the bundle represents a single, aligned feature space:
    `bundle["X_raw"]` must have columns ordered exactly as `bundle["feature_names"]`.

    Processing steps
    ----------------
    1) Identify feature types from encoder metadata, if metadata is provided.

       Continuous features:
           type == "numeric" or any feature not marked as categorical/ordinal.

       Categorical features:
           type in {"onehot", "categorical_passthrough"}

       Ordinal features:
           type in {"ordinal", "ordinal_passthrough"}

    2) Continuous-feature preprocessing:
       - percentile capping
       - missing-value imputation using `continuous_impute_strategy`
       - standard scaling

    3) Categorical/ordinal preprocessing:
       - missing-value imputation using mode
       - no capping
       - no scaling

    Parameters
    ----------
    bundle:
        Bundle dictionary containing:
          - "X_raw"
          - "feature_names"

    lower_q, upper_q:
        Percentiles for capping continuous features.

    continuous_impute_strategy:
        Imputation strategy for continuous numeric features.
        Usually "median" or "mean".

    categorical_impute_strategy:
        Imputation strategy for categorical/discrete features.
        Currently only "mode" is supported.

    ordinal_impute_strategy:
        Imputation strategy for ordinal/discrete features such as EDSS.
        Currently only "mode" is supported.

    preproc_key:
        Key under which preprocessing artifacts are stored.

    meta:
        Encoder metadata returned by encode_categorical_and_ordinal(...).
        If None, all features are treated as continuous.

    Returns
    -------
    bundle:
        The same dictionary object, updated in-place.
    """

    # ------------------------------------------------------------------
    # Validate required bundle keys.
    # ------------------------------------------------------------------
    if "X_raw" not in bundle:
        raise KeyError("bundle must contain key 'X_raw'")

    if "feature_names" not in bundle:
        raise KeyError("bundle must contain key 'feature_names'")

    # ------------------------------------------------------------------
    # Validate imputation strategy settings.
    # ------------------------------------------------------------------
    if continuous_impute_strategy not in ("mean", "median", "most_frequent", "constant"):
        raise ValueError(
            "continuous_impute_strategy must be one of "
            "{'mean', 'median', 'most_frequent', 'constant'}."
        )

    if categorical_impute_strategy != "mode":
        raise ValueError(
            "Only categorical_impute_strategy='mode' is currently supported."
        )

    if ordinal_impute_strategy != "mode":
        raise ValueError(
            "Only ordinal_impute_strategy='mode' is currently supported."
        )

    # ------------------------------------------------------------------
    # Extract feature matrix and feature names.
    # ------------------------------------------------------------------
    X_raw = bundle["X_raw"]
    feature_names = list(bundle["feature_names"])

    # Mapping from feature name to column index.
    bundle["feature_name_to_idx"] = {
        name: i for i, name in enumerate(feature_names)
    }

    # Number of features.
    n_features = X_raw.shape[1]

    # ------------------------------------------------------------------
    # If no metadata is provided, treat all columns as continuous.
    # ------------------------------------------------------------------
    if meta is None:
        X_capped, caps_df = cap_outliers_percentile(
            X_raw,
            feature_names,
            lower_q=lower_q,
            upper_q=upper_q,
        )

        X_imputed, imputer = impute_missing_features(
            X_capped,
            strategy=continuous_impute_strategy,
        )

        X_scaled, scaler = standardize_features(X_imputed)

        bundle["X_scaled"] = X_scaled

        bundle[preproc_key] = {
            "feature_names": feature_names,
            "caps_df": caps_df,
            "imputer": imputer,
            "scaler": scaler,
            "cat_ord_imputer": None,
            "lower_q": lower_q,
            "upper_q": upper_q,
            "continuous_impute_strategy": continuous_impute_strategy,
            "categorical_impute_strategy": categorical_impute_strategy,
            "ordinal_impute_strategy": ordinal_impute_strategy,
            "n_features_fit": int(X_raw.shape[1]),
            "skipped_feature_names": [],
            "cat_ord_imputed_feature_names": [],
            "categorical_imputed_feature_names": [],
            "ordinal_imputed_feature_names": [],
        }

        return bundle

    # ------------------------------------------------------------------
    # Identify categorical/ordinal feature indices to skip from continuous
    # capping/scaling.
    # ------------------------------------------------------------------
    ots = meta.get("output_to_source", {})

    categorical_types = (
        "onehot",
        "categorical_passthrough",
    )

    ordinal_types = (
        "ordinal",
        "ordinal_passthrough",
    )

    cat_ord_types = categorical_types + ordinal_types

    categorical_idx: List[int] = []
    ordinal_idx: List[int] = []
    skip_idx: List[int] = []

    for i, name in enumerate(feature_names):
        feature_type = (ots.get(name) or {}).get("type")

        if feature_type in categorical_types:
            categorical_idx.append(i)
            skip_idx.append(i)

        elif feature_type in ordinal_types:
            ordinal_idx.append(i)
            skip_idx.append(i)

    skip_idx_set = set(skip_idx)

    # Continuous feature indices are everything not categorical/ordinal.
    cont_idx = [
        i for i in range(n_features)
        if i not in skip_idx_set
    ]

    # ------------------------------------------------------------------
    # If no categorical/ordinal features were found, run continuous pipeline
    # on all features.
    # ------------------------------------------------------------------
    if not skip_idx:
        X_capped, caps_df = cap_outliers_percentile(
            X_raw,
            feature_names,
            lower_q=lower_q,
            upper_q=upper_q,
        )

        X_imputed, imputer = impute_missing_features(
            X_capped,
            strategy=continuous_impute_strategy,
        )

        X_scaled, scaler = standardize_features(X_imputed)

        bundle["X_scaled"] = X_scaled

        bundle[preproc_key] = {
            "feature_names": feature_names,
            "caps_df": caps_df,
            "imputer": imputer,
            "scaler": scaler,
            "cat_ord_imputer": None,
            "lower_q": lower_q,
            "upper_q": upper_q,
            "continuous_impute_strategy": continuous_impute_strategy,
            "categorical_impute_strategy": categorical_impute_strategy,
            "ordinal_impute_strategy": ordinal_impute_strategy,
            "n_features_fit": int(X_raw.shape[1]),
            "skipped_feature_names": [],
            "cat_ord_imputed_feature_names": [],
            "categorical_imputed_feature_names": [],
            "ordinal_imputed_feature_names": [],
        }

        return bundle

    # ------------------------------------------------------------------
    # Continuous preprocessing on continuous subset only.
    # ------------------------------------------------------------------
    X_cont = X_raw[:, cont_idx]

    feature_names_cont = [
        feature_names[i]
        for i in cont_idx
    ]

    X_cont_capped, caps_df_cont = cap_outliers_percentile(
        X_cont,
        feature_names_cont,
        lower_q=lower_q,
        upper_q=upper_q,
    )

    X_cont_imputed, imputer = impute_missing_features(
        X_cont_capped,
        strategy=continuous_impute_strategy,
    )

    X_cont_scaled, scaler = standardize_features(X_cont_imputed)

    # ------------------------------------------------------------------
    # Recombine continuous scaled columns with categorical/ordinal columns.
    # Categorical/ordinal columns are still untouched at this point.
    # ------------------------------------------------------------------
    X_scaled_full = np.asarray(X_raw, dtype=np.float32).copy()
    X_scaled_full[:, cont_idx] = X_cont_scaled

    # ------------------------------------------------------------------
    # Mode-impute categorical and ordinal columns only.
    # ------------------------------------------------------------------
    X_scaled_full, cat_ord_imputer, cat_ord_idx = impute_categorical_ordinal_mode(
        X_scaled_full,
        feature_names,
        meta,
        categorical_impute_strategy=categorical_impute_strategy,
        ordinal_impute_strategy=ordinal_impute_strategy,
        categorical_types=categorical_types,
        ordinal_types=ordinal_types,
    )
    # ------------------------------------------------------------------
    # Build full caps_df.
    # Skipped categorical/ordinal columns have NaN caps because they were not
    # capped.
    # ------------------------------------------------------------------
    caps_df_full = pd.DataFrame(
        {
            "lower": np.nan,
            "upper": np.nan,
        },
        index=feature_names,
    )

    caps_df_full.loc[feature_names_cont, :] = caps_df_cont.loc[
        feature_names_cont,
        :
    ]

    # ------------------------------------------------------------------
    # Store outputs and fitted preprocessing artifacts.
    # ------------------------------------------------------------------
    bundle["X_scaled"] = X_scaled_full

    bundle[preproc_key] = {
        "feature_names": feature_names,
        "caps_df": caps_df_full,
        "imputer": imputer,
        "scaler": scaler,
        "cat_ord_imputer": cat_ord_imputer,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "continuous_impute_strategy": continuous_impute_strategy,
        "categorical_impute_strategy": categorical_impute_strategy,
        "ordinal_impute_strategy": ordinal_impute_strategy,
        "n_features_fit": int(X_raw.shape[1]),
        "skipped_feature_names": [
            feature_names[i]
            for i in skip_idx
        ],
        "continuous_feature_names": [
            feature_names[i]
            for i in cont_idx
        ],
        "cat_ord_imputed_feature_names": [
            feature_names[i]
            for i in cat_ord_idx
        ],
        "categorical_imputed_feature_names": [
            feature_names[i]
            for i in categorical_idx
        ],
        "ordinal_imputed_feature_names": [
            feature_names[i]
            for i in ordinal_idx
        ],
        "categorical_impute_types": categorical_types,
        "ordinal_impute_types": ordinal_types,
    }

    return bundle
