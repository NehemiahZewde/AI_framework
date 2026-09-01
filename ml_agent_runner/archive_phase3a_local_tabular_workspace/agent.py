"""OpenAI Agents SDK wrapper for the general chat with native file inputs."""

from __future__ import annotations

from typing import Any, Iterable

from agents import Agent, RunConfig, RunContextWrapper, Runner, function_tool, set_default_openai_key
from openai import AsyncOpenAI

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
- Supported tabular files may also have a local pandas representation reserved
  for future ML-framework operations. Call get_active_tabular_workspace when
  the user asks about local table availability, shape, columns, or worksheets.
- Do not claim a local pandas table exists unless the tool confirms it.
- Do not automatically start target selection, preprocessing, or model training.
- Do not force the user into a predefined data or machine-learning workflow.
- Do not claim privileged access to the ai_framework package or external tools.
"""


class ConversationalAgent:
    """Small async facade around one general OpenAI Agents SDK agent."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = _clean_api_key(api_key)
        self.model = model or "gpt-5.5"
        self.agent = Agent[MLAgentContext](
            name="General Conversational Agent",
            model=self.model,
            instructions=GENERAL_CONVERSATIONAL_INSTRUCTIONS,
            tools=[get_active_tabular_workspace],
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
