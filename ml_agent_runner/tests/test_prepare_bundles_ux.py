"""UX and state tests for grouped training/validation configuration."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from agents.tool import FunctionToolResult
from agents.tool_context import ToolContext

from dataset_setup import build_standardized_dataset_setup
from direct_tool_output import (
    DIRECT_OUTPUT_TOOL_NAMES,
    MODEL_SYNTHESIS_TOOL_NAMES,
    resolve_tool_results,
)
from ml_project_state import MLProjectState
from prepare_bundles_tools import (
    configure_external_prepare_bundles,
    get_prepare_bundles_status,
    inspect_prepare_bundles_function_call,
    inspect_step_1_results,
    run_prepare_train_validation_bundles,
    set_prepare_bundles_validation_mode,
    show_prepare_bundles_advanced_settings,
    show_step_1_execution_log,
    start_prepare_bundles_stage,
    update_internal_prepare_bundles,
)
from prepare_bundles_workflow import PREPARE_BUNDLES_DIRECT_TOOL_NAMES
from tabular_workspace import MLAgentContext, TabularWorkspace


FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))


class PrepareBundlesUxTests(unittest.TestCase):
    def test_01_start_stage_shows_one_recommended_configuration_screen(self) -> None:
        state = self._state()

        output, markdown = self._run_tool(start_prepare_bundles_stage, state)

        self.assertEqual(output["prepare_bundles_status"], "awaiting_configuration")
        self.assertEqual(output["validation_mode"], "internal")
        self.assertIn("## Prepare the training and validation data", markdown)
        self.assertIn("### Primary settings", markdown)
        self.assertIn("**Internal split (Recommended)**", markdown)
        self.assertIn("**20%**", markdown)
        self.assertIn("### Advanced settings", markdown)
        self.assertIn("progress-history storage are enabled", markdown)
        self.assertNotIn("| Progress messages |", markdown)
        self.assertNotIn("Internal outcome name", markdown)
        self.assertIn("Use the recommended settings", markdown)
        self.assertIn("I have a separate validation dataset", markdown)
        self.assertIn("Use all current data without a separate final-validation set", markdown)
        self.assertNotIn("Continue without validation data", markdown)
        self.assertNotIn("Technical summary", markdown)
        self.assertNotIn("prepare_train_validation_bundles", markdown)
        self.assertNotIn("split_kwargs", markdown)
        self.assertNotIn("bundle", markdown.casefold())

    def test_02_accept_recommendations_shows_one_resolved_table_and_confirmation(self) -> None:
        state = self._started_state()

        output, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(),
        )

        self.assertEqual(output["prepare_bundles_status"], "awaiting_final_confirmation")
        self.assertIn("## Step 1 — Review raw data preparation", markdown)
        self.assertIn("| Area | Input or decision | Resolved value | Source |", markdown)
        self.assertIn("| Validation | Validation approach | Internal split |", markdown)
        self.assertIn("| Validation | Training proportion | 80% | Derived |", markdown)
        self.assertIn("| Validation | Final-validation proportion | 20% |", markdown)
        self.assertIn("### Exact Step 1 function call", markdown)
        self.assertIn("'validation_size': 0.2", markdown)
        self.assertIn("validation_kwargs=None", markdown)
        self.assertIn("**Run Step 1 using this configuration?**", markdown)
        self.assertIsNone(state.train_bundle)
        self.assertIsNone(state.validation_bundle)
        self.assertIsNone(state.prep_meta)

    def test_02b_show_advanced_settings_is_detailed_and_read_only(self) -> None:
        state = self._started_state()
        before = copy.deepcopy(state.prepare_bundles.__dict__)

        output, markdown = self._run_tool(
            show_prepare_bundles_advanced_settings,
            state,
        )

        self.assertEqual(state.prepare_bundles.__dict__, before)
        self.assertTrue(output["show_advanced_settings"])
        self.assertIn("## Advanced settings", markdown)
        self.assertIn("| Progress messages | `enabled` | Enabled |", markdown)
        self.assertIn("| Internal outcome name | `target_name` | `target` |", markdown)

    def test_02c_review_is_versioned_and_regenerated_without_execution(self) -> None:
        state = self._started_state()
        first, _ = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(),
        )
        first_version = first["step_1_review_version"]

        second, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(random_state="100"),
        )

        self.assertEqual(second["step_1_review_version"], first_version + 1)
        self.assertEqual(state.prepare_bundles.step_1_review_status, "awaiting_confirmation")
        self.assertEqual(
            state.prepare_bundles.reviewed_prepare_bundles_config["split_kwargs"]["random_state"],
            100,
        )
        self.assertIn("'random_state': 100", markdown)
        self.assertIsNone(state.train_bundle)
        self.assertIsNone(state.validation_bundle)
        self.assertIsNone(state.prep_meta)

    def test_02d_numeric_target_keys_keep_float_encoded_values(self) -> None:
        state = self._numeric_state()
        self._run_tool(start_prepare_bundles_stage, state)

        _, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(),
        )

        self.assertIn("`1.0 → 0.0`", markdown)
        self.assertIn("`2.0 → 1.0`", markdown)
        self.assertIn("1.0: 0.0", markdown)
        self.assertIn("2.0: 1.0", markdown)

    def test_03_partial_update_preserves_unmentioned_settings(self) -> None:
        state = self._started_state()
        updates = self._updates(validation_size="25%", random_state="100")

        output, markdown = self._run_tool(update_internal_prepare_bundles, state, updates)

        self.assertEqual(
            state.prepare_bundles.split_kwargs,
            {"validation_size": 0.25, "random_state": 100, "stratify": True},
        )
        self.assertEqual(state.prepare_bundles.progress_kwargs["enabled"], True)
        self.assertIn("| Validation | Training proportion | 75% | Derived |", markdown)
        self.assertIn("| Validation | Final-validation proportion | 25% | User selected |", markdown)
        self.assertIn("| Validation | Random state | 100 | User selected |", markdown)
        self.assertIn("'validation_size': 0.25", markdown)
        self.assertIn("'random_state': 100", markdown)
        self.assertEqual(output["split_kwargs"]["stratify"], True)

    def test_04_multiple_updates_are_validated_and_applied_together(self) -> None:
        state = self._started_state()

        _, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(
                validation_size="30%",
                random_state="7",
                stratify="disabled",
            ),
        )

        self.assertEqual(
            state.prepare_bundles.split_kwargs,
            {"validation_size": 0.30, "random_state": 7, "stratify": False},
        )
        self.assertIn("| Validation | Final-validation proportion | 30% | User selected |", markdown)
        self.assertIn("| Validation | Stratification | Disabled | User selected |", markdown)

    def test_04b_changed_advanced_setting_is_shown_in_detail(self) -> None:
        state = self._started_state()

        _, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(progress_enabled="disabled"),
        )

        self.assertFalse(state.prepare_bundles.progress_kwargs["enabled"])
        self.assertIn("| Reporting | Progress messages | Disabled | User selected |", markdown)
        self.assertIn("| Target | Internal target name | `target` | Framework default |", markdown)

    def test_05_invalid_update_preserves_previous_valid_state(self) -> None:
        state = self._started_state()
        prior_split = dict(state.prepare_bundles.split_kwargs or {})

        output, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(validation_size="150%"),
        )

        self.assertFalse(output["ok"])
        self.assertEqual(state.prepare_bundles.split_kwargs, prior_split)
        self.assertEqual(state.prepare_bundles.status, "awaiting_configuration")
        self.assertIn("greater than 0 and less than 1", markdown)
        self.assertIn("previous valid settings were preserved", markdown)

    def test_06_external_validation_waits_for_a_separate_upload(self) -> None:
        state = self._started_state()

        output, markdown = self._run_tool(
            set_prepare_bundles_validation_mode,
            state,
            {"validation_mode": "separate validation dataset"},
        )

        self.assertEqual(output["validation_mode"], "external")
        self.assertEqual(output["prepare_bundles_status"], "awaiting_external_data")
        self.assertIsNone(state.prepare_bundles.split_kwargs)
        self.assertIsNone(state.train_bundle)
        self.assertIn("Please attach the separate validation dataset", markdown)
        self.assertNotIn("validation_kwargs", markdown)

        state.attach_external_validation_workspace(self._external_workspace())
        configured, review = self._run_tool(
            configure_external_prepare_bundles,
            state,
            {
                "external_target_col": "Outcome",
                "target_name": "target",
                "progress_enabled": True,
                "show_output_shapes": True,
                "return_progress_log": True,
                "show_progress": True,
            },
        )
        self.assertEqual(configured["prepare_bundles_status"], "awaiting_final_confirmation")
        self.assertEqual(configured["validation_mode"], "external")
        self.assertEqual(configured["validation_kwargs_summary"]["X_shape"][0], 4)
        self.assertIn("| Validation | Validation approach | Separate external validation dataset |", review)
        self.assertIn("| Validation | External outcome column | `Outcome` | User selected |", review)
        self.assertIn('"X": external_validation_X', review)
        self.assertIn("split_kwargs=None", review)

        completed, execution = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )
        self.assertEqual(completed["result_validation_mode"], "provided_validation")
        self.assertEqual(completed["validation_row_count"], 4)
        self.assertIn("[OK] Resolve provided validation inputs", execution)
        self.assertIn("[OK] Align provided validation features", execution)

    def test_07_framework_supported_training_only_path_is_available(self) -> None:
        state = self._started_state()

        output, markdown = self._run_tool(
            set_prepare_bundles_validation_mode,
            state,
            {"validation_mode": "use all current data without a separate final-validation set"},
        )

        self.assertEqual(output["validation_mode"], "none")
        self.assertEqual(output["prepare_bundles_status"], "awaiting_final_confirmation")
        self.assertEqual(state.prepare_bundles.split_kwargs["validation_size"], 0.0)
        self.assertFalse(state.prepare_bundles.split_kwargs["stratify"])
        self.assertIn("## Step 1 — Review raw data preparation", markdown)
        self.assertIn("| Validation | Validation approach | No separate final-validation set |", markdown)
        self.assertIn("| Validation | Data available for model development | 100% | Derived |", markdown)
        self.assertIn("'validation_size': 0.0", markdown)
        self.assertIn("later evaluation strategy", markdown.casefold())
        self.assertNotIn("not recommended", markdown.casefold())
        self.assertNotIn("unsafe", markdown.casefold())

        completed, completion = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )
        self.assertEqual(completed["prepare_bundles_status"], "complete")
        self.assertEqual(completed["training_row_count"], 20)
        self.assertEqual(completed["validation_row_count"], 0)
        self.assertIsNone(state.validation_bundle)
        self.assertEqual(state.prep_meta["validation_mode"], "train_only")
        self.assertIn("Prepare raw train/validation bundles", completion)
        self.assertIn("[OK] Pipeline complete", completion)
        self.assertIn("Step 1 completed successfully", completion)
        self.assertNotIn("| Result | Value |", completion)

    def test_08_status_question_does_not_change_pending_configuration(self) -> None:
        state = self._started_state()
        before = dict(state.prepare_bundles.split_kwargs or {})

        output, markdown = self._run_tool(get_prepare_bundles_status, state)

        self.assertEqual(output["prepare_bundles_status"], "awaiting_configuration")
        self.assertEqual(state.prepare_bundles.split_kwargs, before)
        self.assertIn("Cross-validation folds, including stratified or nested", markdown)

    def test_09_confirmation_runs_the_real_framework_function(self) -> None:
        state = self._started_state()
        self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(random_state="0"),
        )

        output, markdown = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )

        self.assertEqual(output["prepare_bundles_status"], "complete")
        self.assertIsNotNone(state.train_bundle)
        self.assertIsNotNone(state.validation_bundle)
        self.assertIsNotNone(state.prep_meta)
        self.assertEqual(output["training_row_count"], 16)
        self.assertEqual(output["validation_row_count"], 4)
        self.assertIn("Prepare raw train/validation bundles", markdown)
        self.assertIn(">> Resolve training inputs", markdown)
        self.assertIn("[SKIP] Resolve provided validation inputs", markdown)
        self.assertIn("[OK] Pipeline complete", markdown)
        self.assertIn(
            "Step 1 completed successfully. The returned features remain raw; no preprocessing has run.",
            markdown,
        )
        self.assertNotIn("| Result | Value |", markdown)
        self.assertNotIn("Outcome distribution", markdown)
        self.assertNotIn("next step", markdown.casefold())
        self.assertEqual(state.prepare_bundles.status, "complete")
        self.assertEqual(
            state.prepare_bundles.step_1_executed_review_version,
            state.prepare_bundles.step_1_review_version,
        )
        self.assertIsNotNone(state.prepare_bundles.step_1_execution_log)
        self.assertIn("progress_log", state.prep_meta)

    def test_09b_changed_configuration_blocks_execution_and_regenerates_review(self) -> None:
        state = self._started_state()
        first, _ = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(),
        )
        state.prepare_bundles.split_kwargs["random_state"] = 100

        blocked, markdown = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )

        self.assertTrue(blocked["execution_blocked_for_updated_review"])
        self.assertEqual(blocked["step_1_review_version"], first["step_1_review_version"] + 1)
        self.assertEqual(state.prepare_bundles.status, "awaiting_final_confirmation")
        self.assertIn("'random_state': 100", markdown)
        self.assertIn("Run Step 1 using this configuration?", markdown)
        self.assertIsNone(state.train_bundle)
        self.assertIsNone(state.validation_bundle)
        self.assertIsNone(state.prep_meta)

    def test_09bb_same_shaped_data_replacement_invalidates_review(self) -> None:
        state = self._started_state()
        first, _ = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(),
        )
        state.X = state.X.copy()

        blocked, markdown = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )

        self.assertTrue(blocked["execution_blocked_for_updated_review"])
        self.assertEqual(blocked["step_1_review_version"], first["step_1_review_version"] + 1)
        self.assertIn("Run Step 1 using this configuration?", markdown)
        self.assertIsNone(state.train_bundle)

    def test_09c_saved_execution_log_is_read_only(self) -> None:
        state = self._completed_state()
        run_count = state.prepare_bundles.run_count
        train_bundle = state.train_bundle
        validation_bundle = state.validation_bundle

        output, markdown = self._run_tool(show_step_1_execution_log, state)

        self.assertTrue(output["show_saved_execution_log"])
        self.assertIn("### Step 1 execution log", markdown)
        self.assertIn("[OK] Pipeline complete", markdown)
        self.assertEqual(state.prepare_bundles.run_count, run_count)
        self.assertIs(state.train_bundle, train_bundle)
        self.assertIs(state.validation_bundle, validation_bundle)

    def test_09d_structured_result_inspection_does_not_parse_or_rerun_log(self) -> None:
        state = self._completed_state()
        run_count = state.prepare_bundles.run_count

        facts = self._invoke_tool(inspect_step_1_results, state)

        self.assertEqual(facts["result_stratify"], True)
        self.assertEqual(facts["validation_row_count"], 4)
        self.assertEqual(facts["split_kwargs"]["validation_size"], 0.20)
        self.assertEqual(state.prepare_bundles.run_count, run_count)
        self.assertIn(inspect_step_1_results.name, MODEL_SYNTHESIS_TOOL_NAMES)

    def test_09e_duplicate_run_requires_explicit_rerun(self) -> None:
        state = self._completed_state()
        run_count = state.prepare_bundles.run_count
        train_bundle = state.train_bundle

        output, markdown = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )

        self.assertFalse(output["ok"])
        self.assertTrue(output["duplicate_execution"])
        self.assertIn("Rerun Step 1", markdown)
        self.assertEqual(state.prepare_bundles.run_count, run_count)
        self.assertIs(state.train_bundle, train_bundle)

    def test_09f_explicit_rerun_replaces_outputs_only_after_success(self) -> None:
        state = self._completed_state()
        previous_train_bundle = state.train_bundle
        previous_run_count = state.prepare_bundles.run_count

        output, markdown = self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": True},
        )

        self.assertEqual(output["prepare_bundles_status"], "complete")
        self.assertEqual(state.prepare_bundles.run_count, previous_run_count + 1)
        self.assertIsNot(state.train_bundle, previous_train_bundle)
        self.assertIn("Step 1 completed successfully", markdown)

    def test_09g_failed_rerun_preserves_previous_outputs_and_partial_log(self) -> None:
        state = self._completed_state()
        previous_train_bundle = state.train_bundle
        previous_validation_bundle = state.validation_bundle
        previous_prep_meta = state.prep_meta
        previous_executed_version = state.prepare_bundles.step_1_executed_review_version

        def fail_after_progress(**_: object) -> object:
            print("Prepare raw train/validation bundles")
            print("------------------------------------")
            print(">> Resolve training inputs")
            print("[FAIL] Resolve training inputs -> controlled failure")
            raise ValueError("controlled failure")

        with patch(
            "ai_framework.ml_data_preprocessing.prepare_train_validation_bundles",
            side_effect=fail_after_progress,
        ):
            output, markdown = self._run_tool(
                run_prepare_train_validation_bundles,
                state,
                {"allow_rerun": True},
            )

        self.assertFalse(output["ok"])
        self.assertEqual(state.prepare_bundles.status, "failed")
        self.assertEqual(state.prepare_bundles.step_1_execution_error, "controlled failure")
        self.assertIs(state.train_bundle, previous_train_bundle)
        self.assertIs(state.validation_bundle, previous_validation_bundle)
        self.assertIs(state.prep_meta, previous_prep_meta)
        self.assertEqual(
            state.prepare_bundles.step_1_executed_review_version,
            previous_executed_version,
        )
        self.assertIn("[FAIL] Resolve training inputs", markdown)
        self.assertIn("last valid outputs were preserved", markdown)
        self.assertNotIn("completed successfully", markdown)
        self.assertNotIn("Step 2", markdown)

    def test_10_configuration_tools_keep_direct_output_and_valid_schemas(self) -> None:
        self.assertTrue(PREPARE_BUNDLES_DIRECT_TOOL_NAMES <= DIRECT_OUTPUT_TOOL_NAMES)
        self.assertIn(
            inspect_prepare_bundles_function_call.name,
            MODEL_SYNTHESIS_TOOL_NAMES,
        )
        for tool in (
            start_prepare_bundles_stage,
            set_prepare_bundles_validation_mode,
            update_internal_prepare_bundles,
            show_prepare_bundles_advanced_settings,
            show_step_1_execution_log,
            run_prepare_train_validation_bundles,
        ):
            for parameter_schema in tool.params_json_schema.get("properties", {}).values():
                self.assertIsInstance(parameter_schema.get("type"), str)

        state = self._started_state()
        _, markdown = self._run_tool(
            update_internal_prepare_bundles,
            state,
            self._updates(validation_size="25%"),
        )
        self.assertIn("Step 1 — Review raw data preparation", markdown)

        inspection = self._invoke_tool(inspect_prepare_bundles_function_call, state)
        self.assertTrue(inspection["ok"])
        self.assertIn("mdp.prepare_train_validation_bundles(", inspection["resolved_function_call"])

    @staticmethod
    def _updates(**changes: str) -> dict[str, str]:
        values = {
            "validation_size": "",
            "random_state": "",
            "stratify": "",
            "target_name": "",
            "progress_enabled": "",
            "show_output_shapes": "",
            "return_progress_log": "",
            "show_progress": "",
        }
        values.update(changes)
        return values

    def _started_state(self) -> MLProjectState:
        state = self._state()
        self._run_tool(start_prepare_bundles_stage, state)
        return state

    def _completed_state(self) -> MLProjectState:
        state = self._started_state()
        self._run_tool(update_internal_prepare_bundles, state, self._updates())
        self._run_tool(
            run_prepare_train_validation_bundles,
            state,
            {"allow_rerun": False},
        )
        return state

    @staticmethod
    def _state() -> MLProjectState:
        dataframe = pd.DataFrame(
            {
                "Age": [float(value) for value in range(20, 40)],
                "Marker": [float(value) / 10.0 for value in range(20)],
                "Outcome": ["control", "case"] * 10,
            }
        )
        workspace = TabularWorkspace(
            original_file_name="binary_dataset.csv",
            file_extension=".csv",
            content_type="text/csv",
            dataframe=dataframe,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            column_names=list(dataframe.columns),
        )
        state = MLProjectState.from_workspace(workspace)
        state.apply_setup(
            build_standardized_dataset_setup(
                dataframe,
                target_col="Outcome",
                positive_class_value="case",
            )
        )
        return state

    @staticmethod
    def _external_workspace() -> TabularWorkspace:
        dataframe = pd.DataFrame(
            {
                "Age": [41.0, 42.0, 43.0, 44.0],
                "Marker": [2.1, 2.2, 2.3, 2.4],
                "Outcome": ["control", "case", "control", "case"],
            }
        )
        return TabularWorkspace(
            original_file_name="external_validation.csv",
            file_extension=".csv",
            content_type="text/csv",
            dataframe=dataframe,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            column_names=list(dataframe.columns),
        )

    @staticmethod
    def _numeric_state() -> MLProjectState:
        dataframe = pd.DataFrame(
            {
                "Marker": [float(value) for value in range(20)],
                "Outcome": [1.0, 2.0] * 10,
            }
        )
        workspace = TabularWorkspace(
            original_file_name="numeric_target.csv",
            file_extension=".csv",
            content_type="text/csv",
            dataframe=dataframe,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            column_names=list(dataframe.columns),
        )
        state = MLProjectState.from_workspace(workspace)
        state.apply_setup(
            build_standardized_dataset_setup(
                dataframe,
                target_col="Outcome",
                positive_class_value=2.0,
            )
        )
        return state

    def _run_tool(
        self,
        tool: object,
        state: MLProjectState,
        arguments: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], str]:
        output = self._invoke_tool(tool, state, arguments)
        resolution = resolve_tool_results(
            [FunctionToolResult(tool=tool, output=output, run_item=None)]
        )
        self.assertTrue(resolution.is_final_output)
        return output, str(resolution.final_output)

    def _invoke_tool(
        self,
        tool: object,
        state: MLProjectState,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        async def run() -> dict[str, object]:
            payload = json.dumps(arguments or {})
            context = MLAgentContext(
                session_id="prepare-bundles-ux-test",
                ml_project_state=state,
            )
            tool_context = ToolContext(
                context=context,
                tool_name=tool.name,
                tool_call_id=f"call-{tool.name}",
                tool_arguments=payload,
            )
            return await tool.on_invoke_tool(tool_context, payload)

        return asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
