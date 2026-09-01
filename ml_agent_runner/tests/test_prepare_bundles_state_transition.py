"""Regression tests for validation-mode persistence across agent turns."""

from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import json
import unittest

import pandas as pd
from agents import RunContextWrapper
from agents.tool import FunctionToolResult
from agents.tool_context import ToolContext

from agent import _runtime_agent_instructions
from dataset_setup import build_standardized_dataset_setup
from direct_tool_output import resolve_tool_results
from ml_project_state import MLProjectStateManager
from prepare_bundles_tools import (
    set_prepare_bundles_validation_mode,
    start_prepare_bundles_stage,
    update_internal_prepare_bundles,
)
from tabular_workspace import MLAgentContext, TabularWorkspace


class PrepareBundlesStateTransitionTests(unittest.TestCase):
    def test_internal_mode_persists_into_default_configuration_turn(self) -> None:
        asyncio.run(self._run_internal_mode_sequence())

    async def _run_internal_mode_sequence(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Age": [
                    40.0,
                    42.0,
                    44.0,
                    46.0,
                    48.0,
                    60.0,
                    62.0,
                    64.0,
                    66.0,
                    68.0,
                ],
                "Classification": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
            }
        )
        workspace = TabularWorkspace(
            original_file_name="coimbra.csv",
            file_extension=".csv",
            content_type="text/csv",
            dataframe=dataframe,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            column_names=list(dataframe.columns),
        )
        session_id = "internal-mode-regression"
        manager = MLProjectStateManager()
        state = manager.reset_for_workspace(session_id, workspace)
        state.apply_setup(
            build_standardized_dataset_setup(
                dataframe,
                target_col="Classification",
                positive_class_value="2",
            )
        )

        start_context = self._context(manager, session_id, workspace)
        start_output, _ = await self._invoke(
            start_prepare_bundles_stage,
            start_context,
            {},
        )
        self.assertEqual(start_output["prepare_bundles_status"], "awaiting_configuration")
        self.assertEqual(start_output["validation_mode"], "internal")
        self.assertEqual(
            start_output["split_kwargs"],
            {"validation_size": 0.20, "random_state": 42, "stratify": True},
        )
        pending_mode_instructions = _runtime_agent_instructions(
            RunContextWrapper(start_context),
            object(),
        )
        self.assertIn(
            "you MUST call update_internal_prepare_bundles",
            pending_mode_instructions,
        )

        mode_context = self._context(manager, session_id, workspace)
        log_output = io.StringIO()
        with redirect_stdout(log_output):
            mode_output, mode_markdown = await self._invoke(
                set_prepare_bundles_validation_mode,
                mode_context,
                {"validation_mode": "internal split"},
            )

        shared_state = manager.get_or_create(session_id, workspace)
        self.assertIs(shared_state, state)
        self.assertEqual(shared_state.prepare_bundles.validation_mode, "internal")
        self.assertIsNone(shared_state.prepare_bundles.validation_kwargs)
        self.assertEqual(
            shared_state.prepare_bundles.status,
            "awaiting_configuration",
        )
        self.assertEqual(mode_output["validation_mode"], "internal")
        self.assertIn("Prepare the training and validation data", mode_markdown)
        transition_log = log_output.getvalue()
        self.assertIn(f"session_id={session_id}", transition_log)
        self.assertIn(f"state_object_id={id(state)}", transition_log)
        self.assertIn("validation_mode_before=internal", transition_log)
        self.assertIn("validation_mode_after=internal", transition_log)
        self.assertIn("status_before=awaiting_configuration", transition_log)
        self.assertIn("status_after=awaiting_configuration", transition_log)
        pending_defaults_instructions = _runtime_agent_instructions(
            RunContextWrapper(mode_context),
            object(),
        )
        self.assertIn(
            "you MUST call update_internal_prepare_bundles",
            pending_defaults_instructions,
        )

        defaults_context = self._context(manager, session_id, workspace)
        defaults_output, defaults_markdown = await self._invoke(
            update_internal_prepare_bundles,
            defaults_context,
            {
                "target_name": "",
                "validation_size": "",
                "random_state": "",
                "stratify": "",
                "progress_enabled": "",
                "show_output_shapes": "",
                "return_progress_log": "",
                "show_progress": "",
            },
        )
        self.assertTrue(defaults_output["ok"])
        self.assertEqual(
            defaults_output["prepare_bundles_status"],
            "awaiting_final_confirmation",
        )
        self.assertEqual(
            defaults_output["split_kwargs"],
            {
                "validation_size": 0.20,
                "random_state": 42,
                "stratify": True,
            },
        )
        self.assertEqual(defaults_output["step_1_review_status"], "awaiting_confirmation")
        self.assertEqual(defaults_output["step_1_review_version"], 1)
        self.assertIn("Step 1 — Review raw data preparation", defaults_markdown)
        self.assertIn("Run Step 1 using this configuration?", defaults_markdown)
        self.assertNotIn(
            "Select internal validation before configuring split settings",
            defaults_markdown,
        )

    @staticmethod
    def _context(
        manager: MLProjectStateManager,
        session_id: str,
        workspace: TabularWorkspace,
    ) -> MLAgentContext:
        return MLAgentContext(
            session_id=session_id,
            tabular_workspace=workspace,
            ml_project_state=manager.get_or_create(session_id, workspace),
        )

    @staticmethod
    async def _invoke(
        tool: object,
        context: MLAgentContext,
        arguments: dict[str, object],
    ) -> tuple[dict[str, object], str]:
        payload = json.dumps(arguments)
        tool_context = ToolContext(
            context=context,
            tool_name=tool.name,
            tool_call_id=f"call-{tool.name}",
            tool_arguments=payload,
        )
        output = await tool.on_invoke_tool(tool_context, payload)
        resolution = resolve_tool_results(
            [FunctionToolResult(tool=tool, output=output, run_item=None)]
        )
        if not resolution.is_final_output:
            raise AssertionError(f"{tool.name} did not use direct final output")
        return output, str(resolution.final_output)


if __name__ == "__main__":
    unittest.main()
