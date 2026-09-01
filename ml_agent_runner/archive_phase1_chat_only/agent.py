"""OpenAI Agents SDK wrapper for the Phase 1 conversational agent."""

from __future__ import annotations

from typing import Any

from agents import Agent, RunConfig, Runner, set_default_openai_key
from openai import AsyncOpenAI


GENERAL_CONVERSATIONAL_INSTRUCTIONS = """
You are a general-purpose conversational assistant.

- Answer general questions naturally, clearly, and concisely.
- Use prior conversation context to understand follow-up questions, topic
  changes, and references to earlier discussion.
- Acknowledge uncertainty when information is unavailable instead of inventing
  facts.
- Do not claim access to uploaded files, datasets, external tools, or the
  ai_framework package. Those capabilities are not part of this application.
"""


class ConversationalAgent:
    """Small async facade around one general OpenAI Agents SDK agent."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = _clean_api_key(api_key)
        self.model = model or "gpt-5.5"
        self.agent = Agent(
            name="General Conversational Agent",
            model=self.model,
            instructions=GENERAL_CONVERSATIONAL_INSTRUCTIONS,
        )
        self.run_config = RunConfig(
            trace_include_sensitive_data=False,
            tracing_disabled=True,
        )

    async def respond(self, user_message: str, session: Any) -> str:
        """Run one conversational turn against the supplied SDK session."""

        if not self.api_key:
            raise ValueError("An API key is required before running the agent.")

        set_default_openai_key(self.api_key, use_for_tracing=False)
        result = await Runner.run(
            self.agent,
            user_message,
            session=session,
            max_turns=8,
            run_config=self.run_config,
        )
        return str(result.final_output or "I do not have a response for that yet.")


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
