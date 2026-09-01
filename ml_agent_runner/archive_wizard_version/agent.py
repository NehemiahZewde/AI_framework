"""OpenAI Agents SDK wrapper for the Dataset Explorer Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agents import Agent, RunConfig, Runner, set_default_openai_key
from openai import AsyncOpenAI

from dataset_profile import (
    load_and_profile_csv,
    render_dataset_profile_markdown,
    render_draft_setup_markdown,
)
from tools import profile_uploaded_csv


DATASET_EXPLORER_INSTRUCTIONS = """
You are a Dataset Explorer Agent for a healthcare-oriented ML framework.

Version 1 scope:
- Load and profile one uploaded CSV.
- Write a concise first report with a short summary first, followed by the
  detailed profile table.
- Let the Chainlit app ask the target and binary-classification follow-up
  questions.
- Summarize the draft ML setup after confirmation.

Hard limits:
- Do not run model training.
- Do not run feature selection.
- Do not invoke or modify existing ai_framework modules.
- Do not invent dataset facts beyond supplied profile/tool output.
"""


class DatasetExplorerAgent:
    """Small async facade around OpenAI Agents SDK calls."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = _clean_api_key(api_key) or _clean_api_key(os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("ML_AGENT_MODEL", "gpt-5.5")
        self.agent = Agent(
            name="Dataset Explorer Agent",
            model=self.model,
            instructions=DATASET_EXPLORER_INSTRUCTIONS,
            tools=[profile_uploaded_csv],
        )
        self.run_config = RunConfig(
            trace_include_sensitive_data=False,
            tracing_disabled=True,
        )

    async def profile_report(self, csv_path: str | Path) -> str:
        """Use the agent to write the dataset profile report."""

        if not self.api_key or self.run_config is None:
            profile = load_and_profile_csv(csv_path)
            return _without_api_key_note(render_dataset_profile_markdown(profile))

        self._configure_agents_sdk_key()
        prompt = (
            "Use the profile_uploaded_csv tool on this exact file path, then "
            "write a concise Dataset Explorer report. Start with a short "
            "summary containing: rows x columns, likely target column, number "
            "of likely numeric feature columns, number of likely categorical "
            "columns, missingness summary, and ID-like column summary. Then "
            "include the detailed profile table. Do not ask the user to confirm "
            "the target column; the Chainlit app will ask that separately.\n\n"
            f"CSV path: {Path(csv_path)}"
        )
        result = await Runner.run(
            self.agent,
            prompt,
            max_turns=4,
            run_config=self.run_config,
        )
        return str(result.final_output)

    async def setup_summary(self, setup: dict[str, Any]) -> str:
        """Use the agent to summarize a confirmed draft ML setup."""

        if not self.api_key or self.run_config is None:
            return _without_api_key_note(render_draft_setup_markdown(setup))

        self._configure_agents_sdk_key()
        prompt = (
            "Summarize this confirmed draft ML setup. Keep it concise and "
            "explicitly state that no model training or feature selection has "
            f"been run.\n\n{json.dumps(setup, indent=2)}"
        )
        result = await Runner.run(
            self.agent,
            prompt,
            max_turns=2,
            run_config=self.run_config,
        )
        return str(result.final_output)

    def _configure_agents_sdk_key(self) -> None:
        set_default_openai_key(self.api_key, use_for_tracing=False)


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


def _without_api_key_note(markdown: str) -> str:
    return (
        f"{markdown}\n\n"
        "_No OpenAI API key is available for this session, so this deterministic "
        "pandas summary was shown without calling the OpenAI Agents SDK runtime._"
    )
