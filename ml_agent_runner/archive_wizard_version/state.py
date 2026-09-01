"""Session-state keys and containers for the Chainlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CSV_FILE_NAME_KEY = "csv_file_name"
DATASET_PROFILE_KEY = "dataset_profile"
DRAFT_SETUP_KEY = "draft_ml_setup"
OPENAI_API_KEY_SESSION_KEY = "openai_api_key"
STANDARDIZED_DATASET_SETUP_KEY = "standardized_dataset_setup"
FEATURE_GROUPS_KEY = "feature_groups"
CONFIRMED_FEATURE_GROUPS_KEY = "confirmed_feature_groups"


@dataclass
class DatasetExplorerState:
    """In-memory state for one Chainlit dataset exploration session."""

    csv_file_name: str | None = None
    openai_api_key: str | None = None
    profile: dict[str, Any] | None = None
    draft_setup: dict[str, Any] | None = None
    standardized_dataset_setup: dict[str, Any] | None = None
    feature_groups: dict[str, Any] | None = None
    confirmed_feature_groups: dict[str, Any] | None = None
    messages: list[str] = field(default_factory=list)
