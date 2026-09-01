"""Regression tests for the condensed, evidence-aware prediction-target flow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
import unittest

import pandas as pd
from agents.tool import FunctionToolResult
from agents.tool_context import ToolContext

from app import _render_local_workspace_success, _render_startup_greeting
from direct_tool_output import (
    DIRECT_OUTPUT_TOOL_NAMES,
    MODEL_SYNTHESIS_TOOL_NAMES,
    resolve_tool_results,
)
from ml_project_state import MLProjectState
from tabular_workspace import MLAgentContext, TabularWorkspace
from target_setup_tools import (
    confirm_prediction_target_setup,
    revise_prediction_target_proposal,
    start_prediction_target_setup,
)
from target_setup_workflow import TARGET_SETUP_DIRECT_TOOL_NAMES


RUNNER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RUNNER_ROOT.parents[1]


class PredictionTargetSetupTests(unittest.TestCase):
    def test_01_production_runner_has_no_dataset_specific_target_mapping(self) -> None:
        forbidden_terms = (
            "breast_cancer_coimbra",
            "coimbra",
            "healthy control",
            "breast cancer",
        )
        production_files = [
            path
            for path in RUNNER_ROOT.glob("*.py")
            if path.name != "__init__.py"
        ]
        for path in production_files:
            source = path.read_text(encoding="utf-8").casefold()
            for term in forbidden_terms:
                self.assertNotIn(term.casefold(), source, path.name)
            self.assertIsNone(
                re.search(r"if\s+.*dataset_name.*classification", source),
                path.name,
            )

    def test_api_key_success_greeting_identifies_the_framework_assistant(self) -> None:
        greeting = _render_startup_greeting()

        self.assertIn("## Your AI Framework Assistant is ready", greeting)
        self.assertIn(
            "Upload a dataset to begin, or ask a question about data preparation, modeling, or the AI framework.",
            greeting,
        )
        self.assertNotIn("What would you like to talk about?", greeting)

    def test_02_documented_metadata_supports_one_combined_proposal(self) -> None:
        dataframe = self._coimbra_dataframe()
        state = self._state(
            dataframe,
            "documented_dataset.csv",
            metadata={
                "target_name": "Classification",
                "class_descriptions": {
                    1.0: "healthy control",
                    2.0: "breast cancer",
                },
                "class_description_source": "dataset_metadata",
            },
        )

        output, markdown = self._run_tool(start_prediction_target_setup, state)

        self.assertEqual(output["target_setup_status"], "awaiting_confirmation")
        self.assertEqual(output["proposed_target_column"], "Classification")
        self.assertEqual(output["proposed_positive_class"], "2.0")
        self.assertEqual(output["class_description_source"], "dataset_metadata")
        self.assertEqual(output["positive_class_source"], "dataset_metadata")
        self.assertIn("Here is the recommended setup", markdown)
        self.assertIn("Reply `Continue`", markdown)
        self.assertNotIn("Should this be configured", markdown)

    def test_03_generic_coimbra_upload_does_not_invent_numeric_meanings(self) -> None:
        dataframe = self._coimbra_dataframe()
        workspace = self._workspace(dataframe, "breast_cancer_coimbra.csv")
        state = MLProjectState.from_workspace(workspace)

        output, markdown = self._run_tool(start_prediction_target_setup, state)
        combined_response = (
            f"{_render_local_workspace_success(workspace, False)}\n\n{markdown}"
        )

        self.assertEqual(output["proposed_target_column"], "Classification")
        self.assertEqual(output["target_values"], ["1.0", "2.0"])
        self.assertIsNone(output["proposed_positive_class"])
        self.assertEqual(output["class_description_source"], "unknown")
        self.assertIn("Loaded `breast_cancer_coimbra.csv`", combined_response)
        self.assertIn("- 116 rows", combined_response)
        self.assertIn("- 10 columns", combined_response)
        self.assertIn("## Let's define what the model should predict", combined_response)
        self.assertIn("`Classification` appears to be", combined_response)
        self.assertIn("Because this column has two possible outcomes", combined_response)
        self.assertIn(
            "What does each value mean, and which one should be considered the positive outcome?",
            combined_response,
        )
        self.assertIn("`1.0` means control", combined_response)
        self.assertIn("`2.0` should be positive", combined_response)
        for forbidden_phrase in (
            "controlled ML setup operations",
            "exactly two non-null values",
            "compatible with the framework",
            "bundle",
            "internal state",
        ):
            self.assertNotIn(forbidden_phrase.casefold(), combined_response.casefold())
        self.assertNotIn("healthy", markdown.casefold())
        self.assertNotIn("cancer", markdown.casefold())

    def test_04_unknown_numeric_labels_require_positive_value_then_map_to_floats(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Age": [45, 51, 37, 62],
                "Marker": [2.1, 3.4, 1.7, 4.2],
                "Outcome": [10, 20, 10, 20],
            }
        )
        state = self._state(dataframe, "unknown_numeric.csv")
        proposed, proposal_markdown = self._run_tool(start_prediction_target_setup, state)
        self.assertEqual(proposed["target_setup_status"], "awaiting_positive_class")
        self.assertIn("`10.0` means control", proposal_markdown)
        self.assertIn("`20.0` means disease", proposal_markdown)
        self.assertIn("`20.0` should be positive", proposal_markdown)
        self.assertNotIn("`1.0` means", proposal_markdown)
        self.assertNotIn("`2.0` means", proposal_markdown)

        completed, _ = self._run_tool(
            confirm_prediction_target_setup,
            state,
            {
                "target_col": "Outcome",
                "positive_class_value": "20",
                "negative_class_description": "",
                "positive_class_description": "",
                "class_description_source": "unknown",
            },
        )

        self.assertEqual(completed["target_setup_status"], "complete")
        self.assertEqual(state.target_mapping, {10: 0.0, 20: 1.0})
        self.assertTrue(all(type(value) is float for value in state.target_mapping.values()))

    def test_05_self_describing_labels_support_one_semantic_proposal(self) -> None:
        state = self._state(self._string_label_dataframe(), "string_labels.csv")

        proposed, markdown = self._run_tool(start_prediction_target_setup, state)

        self.assertEqual(proposed["target_setup_status"], "awaiting_confirmation")
        self.assertEqual(proposed["proposed_negative_class"], "control")
        self.assertEqual(proposed["proposed_positive_class"], "disease")
        self.assertEqual(proposed["positive_class_source"], "semantic_inference")
        self.assertIn("The labels suggest this interpretation", markdown)
        self.assertNotIn("semantic inference", markdown.casefold())
        self.assertIn("Does this look right?", markdown)

        self._run_tool(
            confirm_prediction_target_setup,
            state,
            {
                "target_col": "Outcome",
                "positive_class_value": "disease",
                "negative_class_description": "control",
                "positive_class_description": "disease",
                "class_description_source": "semantic_inference",
            },
        )
        self.assertEqual(state.target_mapping, {"control": 0.0, "disease": 1.0})

    def test_06_user_correction_overrides_semantic_orientation(self) -> None:
        state = self._state(self._string_label_dataframe(), "string_labels.csv")
        self._run_tool(start_prediction_target_setup, state)

        completed, markdown = self._run_tool(
            confirm_prediction_target_setup,
            state,
            {
                "target_col": "Outcome",
                "positive_class_value": "control",
                "negative_class_description": "disease",
                "positive_class_description": "control",
                "class_description_source": "user_statement",
            },
        )

        self.assertEqual(state.target_mapping, {"disease": 0.0, "control": 1.0})
        self.assertEqual(completed["class_description_source"], "user_statement")
        self.assertIn("`control` -> `1.0`", markdown)

    def test_07_multiple_plausible_targets_require_a_choice(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Age": [30, 40, 50, 60],
                "Biomarker": [1.1, 2.2, 3.3, 4.4],
                "Diagnosis": ["control", "disease", "control", "disease"],
                "Treatment_Response": ["no", "yes", "yes", "no"],
            }
        )
        state = self._state(dataframe, "ambiguous.csv")

        proposed, markdown = self._run_tool(start_prediction_target_setup, state)
        self.assertEqual(proposed["target_setup_status"], "awaiting_target_choice")
        self.assertCountEqual(
            proposed["candidate_columns"],
            ["Diagnosis", "Treatment_Response"],
        )
        self.assertIn("Which one should the model predict?", markdown)

        revised, _ = self._run_tool(
            revise_prediction_target_proposal,
            state,
            {
                "target_col": "Diagnosis",
                "positive_class_value": "",
                "negative_class_description": "",
                "positive_class_description": "",
                "evidence_source": "user_statement",
                "evidence_reason": "The user selected Diagnosis.",
            },
        )
        self.assertEqual(revised["proposed_target_column"], "Diagnosis")
        self.assertEqual(revised["target_values"], ["control", "disease"])

    def test_08_non_binary_target_blocks_setup(self) -> None:
        dataframe = pd.DataFrame(
            {
                "Feature": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7],
                "Outcome": ["low", "medium", "high", "low", "medium", "high"],
            }
        )
        state = self._state(dataframe, "three_classes.csv")

        output, markdown = self._run_tool(start_prediction_target_setup, state)

        self.assertEqual(output["target_setup_status"], "blocked_non_binary")
        self.assertFalse(output["setup_complete"])
        self.assertEqual(output["target_mapping"], [])
        self.assertIsNone(state.target_mapping)
        self.assertIn("supports two-outcome classification", markdown)

    def test_09_confirmation_automatically_builds_all_setup_objects(self) -> None:
        state = self._state(self._string_label_dataframe(), "complete_flow.csv")
        proposal, proposal_markdown = self._run_tool(
            start_prediction_target_setup,
            state,
        )
        self.assertEqual(proposal["target_setup_status"], "awaiting_confirmation")
        self.assertIn("Reply `Continue`", proposal_markdown)

        completed, markdown = self._run_tool(
            confirm_prediction_target_setup,
            state,
            {
                "target_col": "Outcome",
                "positive_class_value": "disease",
                "negative_class_description": "control",
                "positive_class_description": "disease",
                "class_description_source": "semantic_inference",
            },
        )

        self.assertEqual(completed["target_setup_status"], "complete")
        self.assertIsNotNone(state.df)
        self.assertIsNotNone(state.X)
        self.assertIsNotNone(state.y)
        self.assertEqual(state.feature_names, ["Age", "Marker"])
        self.assertIsNotNone(state.metadata)
        self.assertIsNotNone(state.target_mapping)
        self.assertNotIn("Should I now create", markdown)
        self.assertIn("## Prediction target confirmed", markdown)
        self.assertIn("- **Outcome column:** `Outcome`", markdown)
        self.assertIn("- **Negative outcome:** `control`", markdown)
        self.assertIn("- **Positive outcome:** `disease`", markdown)
        self.assertIn("The prediction target is now set", markdown)
        self.assertIn("The next stage will prepare the training and validation data", markdown)
        self.assertNotIn("Classification", markdown)
        self.assertNotIn("healthy control", markdown)
        self.assertNotIn("breast cancer", markdown)
        self.assertNotIn("outcome values", markdown)
        self.assertEqual(state.prepare_bundles.status, "not_started")

    def test_completion_uses_user_supplied_meanings_without_another_confirmation(self) -> None:
        state = self._state(self._coimbra_dataframe(), "generic_upload.csv")
        self._run_tool(start_prediction_target_setup, state)

        completed, markdown = self._run_tool(
            confirm_prediction_target_setup,
            state,
            {
                "target_col": "Classification",
                "positive_class_value": "2.0",
                "negative_class_description": "healthy control",
                "positive_class_description": "breast cancer",
                "class_description_source": "user_statement",
            },
        )

        self.assertEqual(completed["target_setup_status"], "complete")
        self.assertEqual(state.target_mapping, {1.0: 0.0, 2.0: 1.0})
        self.assertIn("## Prediction target confirmed", markdown)
        self.assertIn("- **Outcome column:** `Classification`", markdown)
        self.assertIn("- **Negative outcome:** `1.0` - healthy control", markdown)
        self.assertIn("- **Positive outcome:** `2.0` - breast cancer", markdown)
        self.assertIn("### Model encoding", markdown)
        self.assertIn("- `1.0` -> `0.0`", markdown)
        self.assertIn("- `2.0` -> `1.0`", markdown)
        self.assertIn("- 116 rows", markdown)
        self.assertIn("- 9 input features", markdown)
        self.assertIn(
            "No training/validation split, preprocessing, feature selection, or model training has run yet.",
            markdown,
        )
        self.assertIn(
            "The next stage will prepare the training and validation data.",
            markdown,
        )
        self.assertNotIn("Should I", markdown)
        self.assertNotIn("Prediction outcome confirmed", markdown)
        self.assertNotIn("outcome values", markdown)
        self.assertNotIn("ready for the next preparation stage", markdown)
        self.assertNotIn("bundle", markdown.casefold())

    def test_10_target_mutations_use_direct_output_without_model_synthesis(self) -> None:
        self.assertTrue(TARGET_SETUP_DIRECT_TOOL_NAMES <= DIRECT_OUTPUT_TOOL_NAMES)
        self.assertTrue(TARGET_SETUP_DIRECT_TOOL_NAMES.isdisjoint(MODEL_SYNTHESIS_TOOL_NAMES))
        for tool in (
            start_prediction_target_setup,
            revise_prediction_target_proposal,
            confirm_prediction_target_setup,
        ):
            properties = tool.params_json_schema.get("properties", {})
            for parameter_schema in properties.values():
                self.assertIsInstance(parameter_schema.get("type"), str)

        state = self._state(self._string_label_dataframe(), "direct.csv")
        self._run_tool(start_prediction_target_setup, state)
        output, _ = self._invoke(
            confirm_prediction_target_setup,
            state,
            {
                "target_col": "Outcome",
                "positive_class_value": "disease",
                "negative_class_description": "control",
                "positive_class_description": "disease",
                "class_description_source": "semantic_inference",
            },
        )
        resolution = resolve_tool_results(
            [
                FunctionToolResult(
                    tool=confirm_prediction_target_setup,
                    output=output,
                    run_item=None,
                )
            ]
        )
        self.assertTrue(resolution.is_final_output)

    @staticmethod
    def _string_label_dataframe() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Age": [45, 51, 37, 62],
                "Marker": [2.1, 3.4, 1.7, 4.2],
                "Outcome": ["control", "disease", "control", "disease"],
            }
        )

    @staticmethod
    def _coimbra_dataframe() -> pd.DataFrame:
        csv_path = (
            PROJECT_ROOT
            / "AI_framework_test"
            / "breast_cancer_coimbra"
            / "breast_cancer_coimbra.csv"
        )
        return pd.read_csv(csv_path)

    @staticmethod
    def _state(
        dataframe: pd.DataFrame,
        file_name: str,
        metadata: dict[str, object] | None = None,
    ) -> MLProjectState:
        workspace = PredictionTargetSetupTests._workspace(
            dataframe,
            file_name,
            metadata,
        )
        return MLProjectState.from_workspace(workspace)

    @staticmethod
    def _workspace(
        dataframe: pd.DataFrame,
        file_name: str,
        metadata: dict[str, object] | None = None,
    ) -> TabularWorkspace:
        return TabularWorkspace(
            original_file_name=file_name,
            file_extension=".csv",
            content_type="text/csv",
            dataframe=dataframe,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            column_names=[str(column) for column in dataframe.columns],
            dataset_metadata=metadata,
        )

    def _run_tool(
        self,
        tool: object,
        state: MLProjectState,
        arguments: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], str]:
        output, resolution = self._invoke(tool, state, arguments)
        self.assertTrue(resolution.is_final_output)
        return output, str(resolution.final_output)

    @staticmethod
    def _invoke(
        tool: object,
        state: MLProjectState,
        arguments: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], object]:
        async def run() -> dict[str, object]:
            payload = json.dumps(arguments or {})
            context = MLAgentContext(
                session_id="prediction-target-test",
                ml_project_state=state,
            )
            tool_context = ToolContext(
                context=context,
                tool_name=tool.name,
                tool_call_id=f"call-{tool.name}",
                tool_arguments=payload,
            )
            return await tool.on_invoke_tool(tool_context, payload)

        output = asyncio.run(run())
        resolution = resolve_tool_results(
            [FunctionToolResult(tool=tool, output=output, run_item=None)]
        )
        return output, resolution


if __name__ == "__main__":
    unittest.main()
