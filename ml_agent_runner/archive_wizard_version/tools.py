"""OpenAI Agents SDK tools for the Dataset Explorer Agent."""

from __future__ import annotations

import json

from agents import function_tool

from dataset_profile import load_and_profile_csv


@function_tool
def profile_uploaded_csv(file_path: str) -> str:
    """Load one CSV file and return its dataset profile as JSON."""

    profile = load_and_profile_csv(file_path)
    return json.dumps(profile, indent=2)
