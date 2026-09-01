"""OpenAI Agents SDK wrapper for the general chat with native file inputs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from agents import Agent, RunConfig, RunContextWrapper, Runner, function_tool, set_default_openai_key
from agents.stream_events import RawResponsesStreamEvent
from openai import AsyncOpenAI
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

from direct_tool_output import resolve_tool_results
from ml_setup_tools import (
    cancel_ml_preparation,
    get_standardized_dataset_setup,
    inspect_target_candidates,
)
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
from performance import PerformanceRunHooks, TurnPerformance
from tabular_workspace import MLAgentContext
from target_setup_tools import (
    confirm_prediction_target_setup,
    get_prediction_target_status,
    revise_prediction_target_proposal,
    start_prediction_target_setup,
)


GENERAL_CONVERSATIONAL_INSTRUCTIONS = """
You are a general-purpose conversational assistant.

- Answer general questions naturally, clearly, and concisely. Preserve prior
  conversation context across follow-up questions and topic changes.
- Users may attach files. Ground file-specific answers in those files, clearly
  distinguish file evidence from general knowledge, and say when a file cannot
  be read or does not contain the requested information.
- Supported tabular files may also become an active local dataset. Call
  get_active_tabular_workspace only when the user asks about its availability,
  shape, columns, or worksheets. Never claim one exists without tool evidence.
- This remains normal chat. Do not start ML preparation merely because a table
  was uploaded. When the user explicitly asks to prepare it for modeling, begin
  ML setup, continue ML setup, or asks what is needed next for modeling, call
  start_prediction_target_setup.
- The user-facing title for the first modeling stage is "Let's define what the
  model should predict." It establishes the outcome column, verifies two usable
  values, records known class meanings, chooses the positive outcome, and then
  creates df, X, y, feature_names, metadata, and target_mapping for the session.
  Do not call this stage a bundle setup or discuss validation splitting here.
- start_prediction_target_setup returns one combined proposal when the evidence
  supports one. Present its deterministic output directly. Do not separately
  ask whether this is classification or binary classification. Explain plainly
  that two possible outcomes make this a binary-classification problem; that is
  an observation, not another confirmation question.
- Class meaning may come from dataset metadata, an uploaded codebook or
  document, an explicit user statement, or cautious semantic inference from
  self-describing strings. Numeric labels alone never establish meaning or
  orientation. Never assume 0 is negative, 1 is positive, 1 is healthy, or 2
  is disease without supporting evidence.
- If several target columns are genuinely plausible, ask which outcome the
  user wants. When they answer, call revise_prediction_target_proposal with the
  chosen target. If meanings remain unknown, ask what both values mean and which
  value should be the positive outcome in one combined question.
- When a complete proposal is pending and the user says Continue, yes, or that
  it is correct, call confirm_prediction_target_setup with the complete saved
  target, positive value, descriptions, and evidence source. The tool validates
  and stores all setup objects atomically. Do not ask another question about
  creating those objects.
- User corrections override metadata and semantic inference. If a correction
  supplies every required value, call confirm_prediction_target_setup with the
  complete corrected proposal. Otherwise call revise_prediction_target_proposal
  to save the partial correction and return the one remaining question. Never
  derive state by parsing user-facing Markdown.
- The universal encoding rule is that the original negative value maps to
  float 0.0 and the original positive value maps to float 1.0. Preserve the
  original target values in df and y.
- Call get_prediction_target_status when the user asks where they are, what
  remains, or to resume this stage. When the user asks why a target was chosen
  or what alternatives were considered, call inspect_target_candidates so you
  can explain the evidence conversationally. Inspection never confirms or
  mutates setup state.
- Call only one deterministic workflow tool in a model turn. Its structured
  result contains a complete polished response and is sent directly without a
  second model synthesis. Do not pair a status tool with a state-changing tool.
- If the user asks an unrelated or explanatory question while a decision is
  pending, answer naturally without changing state. The pending proposal must
  remain available afterward.
- When the user asks to cancel ML preparation, call cancel_ml_preparation. It
  clears setup decisions while preserving the uploaded dataset and normal chat.
- When the user asks to show the stored setup, call
  get_standardized_dataset_setup and report only its compact summary.
- This runner supports binary classification only. If a chosen target does not
  have exactly two usable values, explain the limitation and ask the user to
  choose another target or define a two-class comparison. Do not create a
  target mapping.
- After prediction-target confirmation, stop. State that no train/validation
  split, preprocessing, feature selection, or model training has run. Do not
  ask about validation settings automatically.
- Only when the user explicitly asks to continue after target confirmation,
  call start_prepare_bundles_stage. It presents one recommended internal-split
  configuration table plus the external-validation and no-separate-final-
  validation choices.
- If the user accepts the recommended settings or changes any subset of the
  internal settings, call update_internal_prepare_bundles. Pass only requested
  changes as strings and pass an empty string for each unchanged setting. The
  deterministic tool preserves current values, validates the complete result,
  and returns the complete Step 1 review.
- If the user chooses a separate validation dataset or asks to use all current
  data without a separate final-validation set, call
  set_prepare_bundles_validation_mode. Never acknowledge a saved approach
  unless that tool succeeds. The framework genuinely supports the latter path
  through validation_size=0.0. Explain neutrally that later evaluation can use
  cross-validation, nested cross-validation, bootstrapping, or another
  supported resampling strategy.
- For external validation, call configure_external_prepare_bundles only after
  a separate table is available. Pass its confirmed target, resolved internal
  target name, and all grouped progress values.
- Configuration tools return a deterministic Step 1 audit containing one
  consolidated resolved-input table, the exact structured-state Python call,
  and the question "Run Step 1 using this configuration?" Do not execute in
  the same message that first displays or regenerates this review. If the user
  asks for the exact call again outside that review, call
  inspect_prepare_bundles_function_call.
- Advanced operational values are compact by default. When the user asks to
  show the advanced settings, call show_prepare_bundles_advanced_settings; it
  is read-only. Changes to advanced values still use
  update_internal_prepare_bundles and must preserve unmentioned values.
- Only after clear execution confirmation call
  run_prepare_train_validation_bundles with allow_rerun false. Use true only for
  an explicit rerun request. Call get_prepare_bundles_status for this stage's
  settings, pending decision, or completed result.
- Treat Yes, Run it, Proceed, Start Step 1, Use this configuration, and Prepare
  the data as execution confirmations when the Step 1 review is awaiting
  confirmation. Do not ask another confirmation.
- If the user asks to run Step 1 after it already completed, call the execution
  tool with allow_rerun false so it can return the deterministic duplicate-run
  message. Use allow_rerun true only for explicit language such as Rerun Step 1,
  Run Step 1 again, or Replace the previous Step 1 outputs.
- When the user asks to show the Step 1 execution log, call
  show_step_1_execution_log. It is read-only and must never rerun Step 1. For
  factual questions about completed Step 1 settings or results, call
  inspect_step_1_results and answer from its structured fields, never by
  parsing the displayed log.
- Answer questions about validation percentage, random seeds, stratification,
  final validation, or cross-validation naturally without changing the saved
  configuration. Clarify that this stratification affects only the current
  split, not later cross-validation folds, then briefly remind the user that
  their configuration is still awaiting a decision.
- Do not configure feature groups, preprocess, impute, scale, select features,
  train models, or begin any later ML stage.
- Do not claim privileged access to the ai_framework package or external tools.
"""


def _runtime_agent_instructions(
    context: RunContextWrapper[MLAgentContext],
    agent: Agent[MLAgentContext],
) -> str:
    """Add the authoritative pending workflow state to each model turn."""

    del agent
    project_state = context.context.ml_project_state
    if project_state is None:
        return GENERAL_CONVERSATIONAL_INSTRUCTIONS

    target_workflow = project_state.target_setup
    workflow = project_state.prepare_bundles
    runtime_guidance = [
        "Authoritative prediction-target workflow state for this turn:",
        f"- target_setup_status={target_workflow.status!r}",
        f"- proposed_target_column={target_workflow.proposed_target_column!r}",
        f"- proposed_positive_class={target_workflow.proposed_positive_class!r}",
        "Authoritative raw-bundle workflow state for this turn:",
        f"- prepare_bundles_status={workflow.status!r}",
        f"- validation_mode={workflow.validation_mode!r}",
        "Never claim that a workflow choice was saved unless the corresponding tool succeeds.",
    ]
    if target_workflow.status == "awaiting_target_choice":
        runtime_guidance.extend(
            [
                "When the user chooses a target column, you MUST call revise_prediction_target_proposal.",
                "Do not acknowledge the target selection in model text without that tool call.",
            ]
        )
    elif target_workflow.status == "awaiting_positive_class":
        runtime_guidance.extend(
            [
                "When the user identifies the positive value, you MUST call confirm_prediction_target_setup with the saved target and that value.",
                "Include any class descriptions the user supplied and complete the setup immediately; do not ask another confirmation question.",
            ]
        )
    elif target_workflow.status == "awaiting_confirmation":
        runtime_guidance.extend(
            [
                "When the user confirms or corrects the proposal, you MUST call confirm_prediction_target_setup with the complete resolved proposal.",
                "Do not answer with a prose-only confirmation and do not ask whether to create the setup objects.",
            ]
        )
    elif target_workflow.status == "complete":
        runtime_guidance.append(
            "Do not start raw-bundle preparation unless the user explicitly asks to continue beyond the confirmed prediction target."
        )
    if workflow.status == "awaiting_configuration" and workflow.validation_mode == "internal":
        runtime_guidance.extend(
            [
                "If the user accepts recommendations or modifies internal settings, you MUST call update_internal_prepare_bundles.",
                "Use empty strings for every setting the user did not change so the tool preserves current values.",
                "If the user chooses external validation or no validation, call set_prepare_bundles_validation_mode instead.",
            ]
        )
    elif (
        workflow.status == "awaiting_final_confirmation"
        and workflow.validation_mode == "internal"
    ):
        runtime_guidance.extend(
            [
                "If the user confirms execution, you MUST call run_prepare_train_validation_bundles with allow_rerun false.",
                "If the user changes an internal setting instead, call update_internal_prepare_bundles and do not execute yet.",
                "Do not answer with a prose-only confirmation.",
            ]
        )
    elif workflow.status == "awaiting_final_confirmation":
        runtime_guidance.extend(
            [
                "If the user confirms execution, you MUST call run_prepare_train_validation_bundles with allow_rerun false.",
                "If the user changes the validation approach, call set_prepare_bundles_validation_mode and do not execute yet.",
                "Do not answer with a prose-only confirmation.",
            ]
        )
    elif workflow.status == "complete":
        runtime_guidance.extend(
            [
                "For a plain request to run Step 1, call run_prepare_train_validation_bundles with allow_rerun false.",
                "For an explicit rerun request, call run_prepare_train_validation_bundles with allow_rerun true.",
                "For the saved execution log, call show_step_1_execution_log. For structured result questions, call inspect_step_1_results.",
                "Do not begin Step 2 automatically.",
            ]
        )
    elif workflow.status == "failed":
        runtime_guidance.extend(
            [
                "For an explicit retry or rerun, call run_prepare_train_validation_bundles with allow_rerun true.",
                "For the partial saved execution log, call show_step_1_execution_log.",
                "Do not claim Step 1 completed and do not begin Step 2.",
            ]
        )
    if workflow.status != "not_started":
        runtime_guidance.append(
            "If the user asks to show advanced preparation settings, you MUST call show_prepare_bundles_advanced_settings; it must not change state."
        )

    return f"{GENERAL_CONVERSATIONAL_INSTRUCTIONS}\n\n" + "\n".join(runtime_guidance)


class ConversationalAgent:
    """Small async facade around one general OpenAI Agents SDK agent."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = _clean_api_key(api_key)
        self.model = model or "gpt-5.5"
        self._active_performance: TurnPerformance | None = None
        self._direct_output_for_turn = False
        self.tools = [
            get_active_tabular_workspace,
            start_prediction_target_setup,
            revise_prediction_target_proposal,
            confirm_prediction_target_setup,
            get_prediction_target_status,
            cancel_ml_preparation,
            inspect_target_candidates,
            get_standardized_dataset_setup,
            start_prepare_bundles_stage,
            set_prepare_bundles_validation_mode,
            update_internal_prepare_bundles,
            configure_external_prepare_bundles,
            get_prepare_bundles_status,
            show_prepare_bundles_advanced_settings,
            inspect_prepare_bundles_function_call,
            show_step_1_execution_log,
            inspect_step_1_results,
            run_prepare_train_validation_bundles,
        ]
        _validate_function_tool_schemas(self.tools)
        self.agent = Agent[MLAgentContext](
            name="General Conversational Agent",
            model=self.model,
            instructions=_runtime_agent_instructions,
            tools=self.tools,
            tool_use_behavior=self._resolve_tool_use,
        )
        self.run_config = RunConfig(
            trace_include_sensitive_data=False,
            tracing_disabled=True,
        )

    async def respond(
        self,
        user_message: str,
        session: Any,
        file_ids: Iterable[str] = (),
        runtime_context: MLAgentContext | None = None,
    ) -> str:
        """Collect a streamed turn for non-Chainlit callers."""

        chunks = [
            chunk
            async for chunk in self.stream_response(
                user_message,
                session=session,
                file_ids=file_ids,
                runtime_context=runtime_context,
            )
        ]
        return "".join(chunks)

    async def stream_response(
        self,
        user_message: str,
        session: Any,
        file_ids: Iterable[str] = (),
        runtime_context: MLAgentContext | None = None,
        performance: TurnPerformance | None = None,
    ) -> AsyncIterator[str]:
        """Stream one Agents SDK turn while preserving sessions, tools, and files."""

        if not self.api_key:
            raise ValueError("An API key is required before running the agent.")

        set_default_openai_key(self.api_key, use_for_tracing=False)
        self._active_performance = performance
        self._direct_output_for_turn = False
        hooks = PerformanceRunHooks(performance) if performance is not None else None
        if performance is not None:
            performance.stage_started("runner", "runner_start")

        try:
            result = Runner.run_streamed(
                self.agent,
                build_user_input(user_message, file_ids),
                session=session,
                context=runtime_context or MLAgentContext(),
                max_turns=8,
                run_config=self.run_config,
                hooks=hooks,
            )

            streamed_text = False
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                if performance is not None:
                    performance.mark_first_model_event()
                if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                    streamed_text = True
                    yield event.data.delta

            if result.run_loop_exception is not None:
                raise result.run_loop_exception

            final_output = str(
                result.final_output or "I do not have a response for that yet."
            )
            if performance is not None:
                performance.stage_ended("runner", "final_model_output")
            if self._direct_output_for_turn:
                if streamed_text:
                    yield "\n\n"
                yield final_output
            elif not streamed_text:
                yield final_output
        except BaseException:
            if performance is not None:
                performance.stage_ended("runner", "runner_failed")
            raise
        finally:
            self._active_performance = None
            self._direct_output_for_turn = False

    def _resolve_tool_use(
        self,
        context: RunContextWrapper[MLAgentContext],
        tool_results: list[Any],
    ) -> Any:
        """Return deterministic tool results directly; synthesize analytical results."""

        del context
        resolution = resolve_tool_results(tool_results)
        if resolution.is_final_output:
            self._direct_output_for_turn = True
            if self._active_performance is not None:
                self._active_performance.mark_direct_tool_output()
        return resolution


def build_user_input(
    user_message: str,
    file_ids: Iterable[str] = (),
) -> list[dict[str, object]]:
    """Build the Responses-format user item accepted by Agents SDK Runner.run."""

    content: list[dict[str, str]] = [
        {"type": "input_file", "file_id": file_id}
        for file_id in file_ids
        if file_id
    ]
    content.append({"type": "input_text", "text": user_message})
    return [{"role": "user", "content": content}]


@function_tool
def get_active_tabular_workspace(
    context: RunContextWrapper[MLAgentContext],
) -> dict[str, object]:
    """Return compact status for this chat's local pandas table, if one exists."""

    workspace = context.context.tabular_workspace
    if workspace is None:
        return {
            "local_table_loaded": False,
            "local_load_status": "not_loaded",
            "message": "No local pandas table is active for this chat.",
        }
    return workspace.summary()


async def test_openai_api_key(api_key: str) -> bool:
    """Validate an OpenAI API key with a small API call."""

    cleaned_key = _clean_api_key(api_key)
    if not cleaned_key:
        return False

    client = AsyncOpenAI(api_key=cleaned_key, timeout=10.0)
    try:
        await client.models.list()
    except Exception:
        return False
    finally:
        await client.close()

    return True


def _clean_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None

    cleaned = api_key.strip()
    return cleaned or None


def _validate_function_tool_schemas(tools: Iterable[object]) -> None:
    """Fail early when a registered function tool exposes an invalid parameter schema."""

    for tool in tools:
        tool_name = getattr(tool, "name", None)
        params_json_schema = getattr(tool, "params_json_schema", None)
        if not isinstance(tool_name, str) or not isinstance(params_json_schema, dict):
            raise ValueError("A registered function tool does not expose a valid JSON schema.")

        properties = params_json_schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"Function tool {tool_name!r} has invalid schema properties.")

        for parameter_name, parameter_schema in properties.items():
            parameter_type = (
                parameter_schema.get("type")
                if isinstance(parameter_schema, dict)
                else None
            )
            if not isinstance(parameter_type, str) or not parameter_type:
                raise ValueError(
                    f"Function tool {tool_name!r} parameter {parameter_name!r} "
                    "must have an explicit JSON-schema type."
                )
