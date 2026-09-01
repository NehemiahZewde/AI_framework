"""Focused pandas loader for the explicitly supported local tabular formats."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd

from tabular_workspace import TabularWorkspace


@dataclass(frozen=True)
class TabularFormat:
    extension: str
    reader: str
    optional_dependency: str | None = None


SUPPORTED_TABULAR_FORMATS: dict[str, TabularFormat] = {
    ".csv": TabularFormat(".csv", "csv"),
    ".tsv": TabularFormat(".tsv", "tsv"),
    ".xlsx": TabularFormat(".xlsx", "excel", "openpyxl"),
    ".xls": TabularFormat(".xls", "excel", "xlrd"),
    ".json": TabularFormat(".json", "json"),
    ".parquet": TabularFormat(".parquet", "parquet", "pyarrow"),
}


@dataclass
class TabularLoadResult:
    """Compact outcome of attempting local pandas preparation for one file."""

    is_supported_tabular_file: bool
    workspace: TabularWorkspace | None = None
    local_load_status: str = "not_applicable"
    local_load_error: str | None = None


class LocalTabularLoadError(ValueError):
    """Expected local pandas preparation failure safe to show in the UI."""


def load_tabular_workspace(
    file_path: str | Path,
    original_file_name: str,
    content_type: str | None = None,
) -> TabularLoadResult:
    """Attempt to load a supported tabular attachment into a local workspace."""

    extension = Path(original_file_name).suffix.casefold()
    table_format = SUPPORTED_TABULAR_FORMATS.get(extension)
    if table_format is None:
        return TabularLoadResult(is_supported_tabular_file=False)

    try:
        source_path = _validate_source_file(file_path)
        _require_optional_dependency(table_format)
        dataframe, active_sheet_name, sheet_names = _read_table(source_path, table_format)
        dataframe = _validate_dataframe(dataframe)
    except LocalTabularLoadError as exc:
        return TabularLoadResult(
            is_supported_tabular_file=True,
            local_load_status="failed",
            local_load_error=str(exc),
        )
    except Exception:
        return TabularLoadResult(
            is_supported_tabular_file=True,
            local_load_status="failed",
            local_load_error="The file could not be prepared as a local pandas table.",
        )

    workspace = TabularWorkspace(
        original_file_name=original_file_name,
        file_extension=extension,
        content_type=content_type,
        dataframe=dataframe,
        row_count=int(dataframe.shape[0]),
        column_count=int(dataframe.shape[1]),
        column_names=[str(column) for column in dataframe.columns],
        sheet_names=sheet_names,
        active_sheet_name=active_sheet_name,
    )
    return TabularLoadResult(
        is_supported_tabular_file=True,
        workspace=workspace,
        local_load_status="loaded",
    )


def _validate_source_file(file_path: str | Path) -> Path:
    source_path = Path(file_path)
    if not source_path.is_file():
        raise LocalTabularLoadError("The local attachment path could not be accessed.")
    try:
        if source_path.stat().st_size == 0:
            raise LocalTabularLoadError("The local tabular file is empty.")
    except OSError as exc:
        raise LocalTabularLoadError("The local attachment path could not be accessed.") from exc
    return source_path


def _require_optional_dependency(table_format: TabularFormat) -> None:
    dependency = table_format.optional_dependency
    if dependency and importlib.util.find_spec(dependency) is None:
        raise LocalTabularLoadError(
            f"Local pandas preparation requires optional package `{dependency}` "
            f"for {table_format.extension} files."
        )


def _read_table(
    source_path: Path,
    table_format: TabularFormat,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    try:
        if table_format.reader == "csv":
            return pd.read_csv(source_path), None, []
        if table_format.reader == "tsv":
            return pd.read_csv(source_path, sep="\t"), None, []
        if table_format.reader == "json":
            return pd.read_json(source_path), None, []
        if table_format.reader == "parquet":
            return pd.read_parquet(source_path), None, []
        if table_format.reader == "excel":
            return _read_excel(source_path, table_format.optional_dependency)
    except pd.errors.EmptyDataError as exc:
        raise LocalTabularLoadError("The local tabular file is empty.") from exc
    except ImportError as exc:
        dependency = table_format.optional_dependency
        if dependency:
            raise LocalTabularLoadError(
                f"Local pandas preparation requires optional package `{dependency}` "
                f"for {table_format.extension} files."
            ) from exc
        raise LocalTabularLoadError("The file could not be read as a local pandas table.") from exc
    except (BadZipFile, OSError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise LocalTabularLoadError("The file could not be read as a local pandas table.") from exc

    raise LocalTabularLoadError("No local pandas reader is configured for this file type.")


def _read_excel(
    source_path: Path,
    engine: str | None,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    try:
        workbook = pd.ExcelFile(source_path, engine=engine)
    except (BadZipFile, OSError, ValueError, TypeError) as exc:
        raise LocalTabularLoadError("The Excel workbook could not be read.") from exc

    try:
        sheet_names = [str(name) for name in workbook.sheet_names]
        if not sheet_names:
            raise LocalTabularLoadError("The Excel workbook has no readable worksheets.")
        active_sheet_name = sheet_names[0]
        dataframe = pd.read_excel(workbook, sheet_name=active_sheet_name)
        return dataframe, active_sheet_name, sheet_names
    finally:
        workbook.close()


def _validate_dataframe(dataframe: object) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame):
        raise LocalTabularLoadError("The attachment did not produce a pandas DataFrame.")
    if dataframe.shape[1] == 0:
        raise LocalTabularLoadError("The local pandas table has no columns.")
    return dataframe
