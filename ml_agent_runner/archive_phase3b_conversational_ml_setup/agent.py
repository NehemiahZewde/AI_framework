"""OpenAI Agents SDK wrapper for the general chat with native file inputs."""

from __future__ import annotations

from typing import Any, Iterable

from agents import Agent, RunConfig, RunContextWrapper, Runner, function_tool, set_default_openai_key
from openai import AsyncOpenAI

from ml_setup_tools import (
    build_current_standardized_dataset_setup,
    get_current_ml_setup,
    set_positive_class,
    set_target_column,
)
from tabular_workspace import MLAgentContext


GENERAL_CONVERSATIONAL_INSTRUCTIONS = """
You are a general-purpose conversational assistant.

- Answer general questions naturally, clearly, and concisely.
- Use prior conversation context to understand follow-up questions, topic
  changes, and references to earlier discussion.
- Users may optionally attach files. When a file is included in a turn, answer
  questions grounded in that file when its contents support the answer.
- Clearly distinguish file-derived information from general knowledge. Do not
  invent details that are not present in an attached file.
- State when a file cannot be read or does not contain the requested
  information.
- Supported tabular files may also have a local pandas representation for
  controlled standardized binary dataset setup. Call
  get_active_tabular_workspace when the user asks about local table
  availability, shape, columns, or worksheets. Call get_current_ml_setup when
  they ask about target selection or the current local setup.
- Do not claim a local pandas table exists unless the tool confirms it.
- When the user explicitly wants to begin ML setup, use the controlled tools
  instead of inventing table facts. The user must choose the target column:
  call set_target_column only after the user names it. Never guess a target.
- If the chosen target has exactly two non-null values, ask the user which
  displayed value should be positive. Never choose it yourself. After the user
  provides a value, call set_positive_class. Once that succeeds, call
  build_current_standardized_dataset_setup in the same turn and summarize its
  returned shapes, metadata, and float target mapping.
- This setup is binary-classification-only. Do not offer a setup for a target
  with anything other than two non-null values.
- Do not start train/validation splitting, preprocessing, feature selection,
  model training, or ai_framework execution.
- Do not force the user into a predefined data or machine-learning workflow.
- Do not claim privileged access to the ai_framework package or external tools.
"""


class ConversationalAgent:
    """Small async facade around one general OpenAI Agents SDK agent."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = _clean_api_key(api_key)
        self.model = model or "gpt-5.5"
        self.tools = [
            get_active_tabular_workspace,
            get_current_ml_setup,
            set_target_column,
            set_positive_class,
            build_current_standardized_dataset_setup,
        ]
        _validate_function_tool_schemas(self.tools)
        self.agent = Agent[MLAgentContext](
            name="General Conversational Agent",
            model=self.model,
            instructions=GENERAL_CONVERSATIONAL_INSTRUCTIONS,
            tools=self.tools,
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
        """Run one conversation turn, optionally including uploaded OpenAI files."""

        if not self.api_key:
            raise ValueError("An API key is required before running the agent.")

        set_default_openai_key(self.api_key, use_for_tracing=False)
        result = await Runner.run(
            self.agent,
            build_user_input(user_message, file_ids),
            session=session,
            context=runtime_context or MLAgentContext(),
            max_turns=8,
            run_config=self.run_config,
        )
        return str(result.final_output or "I do not have a response for that yet.")


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
