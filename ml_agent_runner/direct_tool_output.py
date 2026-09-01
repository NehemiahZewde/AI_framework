"""Selective final-output policy for deterministic workflow tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agents.agent import ToolsToFinalOutputResult
from agents.tool import FunctionToolResult

from prepare_bundles_workflow import (
    PREPARE_BUNDLES_DIRECT_TOOL_NAMES,
    render_prepare_bundles_output,
)
from target_setup_workflow import (
    TARGET_SETUP_DIRECT_TOOL_NAMES,
    render_prediction_target_output,
)

DIRECT_OUTPUT_TOOL_NAMES = frozenset(
    {
        "get_active_tabular_workspace",
        "cancel_ml_preparation",
        "get_standardized_dataset_setup",
    }
).union(PREPARE_BUNDLES_DIRECT_TOOL_NAMES, TARGET_SETUP_DIRECT_TOOL_NAMES)

MODEL_SYNTHESIS_TOOL_NAMES = frozenset(
    {
        "inspect_target_candidates",
        "inspect_prepare_bundles_function_call",
        "inspect_step_1_results",
    }
)


def resolve_tool_results(
    tool_results: Sequence[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """Stop only when every tool result is deterministic and directly renderable."""

    if not tool_results:
        return ToolsToFinalOutputResult(is_final_output=False, final_output=None)

    tool_names = [result.tool.name for result in tool_results]
    if not all(name in DIRECT_OUTPUT_TOOL_NAMES for name in tool_names):
        return ToolsToFinalOutputResult(is_final_output=False, final_output=None)

    # The instructions ask for one deterministic tool per turn. If a model emits
    # several, the final result reflects the last state transition and avoids a
    # second model call or duplicate user-facing messages.
    final_result = tool_results[-1]
    return ToolsToFinalOutputResult(
        is_final_output=True,
        final_output=render_direct_tool_output(
            final_result.tool.name,
            final_result.output,
        ),
    )


def render_direct_tool_output(tool_name: str, output: Any) -> str:
    """Convert a deterministic structured result into complete Markdown."""

    if not isinstance(output, Mapping):
        return str(output)

    data = dict(output)
    if tool_name in TARGET_SETUP_DIRECT_TOOL_NAMES:
        return render_prediction_target_output(tool_name, data)
    if tool_name in PREPARE_BUNDLES_DIRECT_TOOL_NAMES:
        return render_prepare_bundles_output(tool_name, data)
    if data.get("ok") is False:
        return _render_error(tool_name, data)

    renderers = {
        "get_active_tabular_workspace": _render_workspace,
        "cancel_ml_preparation": _render_cancel,
        "get_standardized_dataset_setup": _render_current_setup,
    }
    renderer = renderers.get(tool_name)
    if renderer is None:
        return _render_generic(data)
    return renderer(data)


def _render_workspace(data: dict[str, Any]) -> str:
    if not data.get("local_table_loaded"):
        return str(data.get("message") or "No local pandas table is active for this chat.")

    lines = [
        "**Active local dataset**",
        "",
        f"- File: {_code(data.get('original_filename'))}",
        f"- Shape: {_shape(data.get('row_count'), data.get('column_count'))}",
        f"- Columns: {_code_list(data.get('column_names'))}",
    ]
    if data.get("active_worksheet"):
        lines.append(f"- Active worksheet: {_code(data.get('active_worksheet'))}")
    return "\n".join(lines)


def _render_current_setup(data: dict[str, Any]) -> str:
    if data.get("setup_status") == "completed" or data.get("standardized_setup_built"):
        return "\n".join(
            ["**Current prediction target setup**", "", *_setup_summary_lines(data)]
        )
    return "The prediction target has not been confirmed yet."


def _render_cancel(data: dict[str, Any]) -> str:
    return "\n".join(
        [
            "**ML preparation cancelled**",
            "",
            "The setup decisions were cleared. The active uploaded dataset and normal conversation remain available.",
        ]
    )


def _render_error(tool_name: str, data: dict[str, Any]) -> str:
    del tool_name
    return "\n".join(
        [
            "**ML preparation**",
            "",
            str(data.get("message") or "That change could not be applied."),
        ]
    )


def _setup_summary_lines(data: dict[str, Any]) -> list[str]:
    lines = [
        f"- `df`: {_shape_from_value(data.get('df_shape'))}",
        f"- `X`: {_shape_from_value(data.get('X_shape'))}",
        f"- `y`: {data.get('y_length', 'unknown')} values",
        f"- `feature_names`: {_code_list(data.get('feature_names'))}",
    ]
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        lines.append("- `metadata`:")
        for key, value in metadata.items():
            lines.append(f"  - `{key}`: {_code(value)}")
    mapping = data.get("target_mapping")
    lines.append("- `target_mapping`:")
    if isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes)):
        for entry in mapping:
            if isinstance(entry, Mapping):
                lines.append(
                    f"  - {_code(entry.get('original_value'))} -> {_code(entry.get('encoded_value'))}"
                )
    else:
        lines.append("  - Not created yet")
    if "prepare_bundles_status" in data:
        lines.append(
            f"- Raw train/final-validation bundles: `{data.get('prepare_bundles_status')}`"
        )
    return lines


def _render_generic(data: dict[str, Any]) -> str:
    message = data.get("message")
    return str(message) if message else "The requested workflow action completed."


def _shape_from_value(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return _shape(value[0], value[1])
    return "not created"


def _shape(rows: Any, columns: Any) -> str:
    return f"{rows} rows x {columns} columns"


def _code_list(values: Any) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return "None"
    return ", ".join(_code(value) for value in values) or "None"


def _code(value: Any) -> str:
    if value is None:
        return "`None`"
    return f"`{str(value).replace('`', '')}`"
