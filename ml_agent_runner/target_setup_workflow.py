"""Evidence-aware proposal and presentation for defining a prediction target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dataset_setup import (
    display_target_value,
    get_unique_non_null_target_values,
    inspect_target_candidates,
    match_target_value,
)


TARGET_SETUP_DIRECT_TOOL_NAMES = frozenset(
    {
        "start_prediction_target_setup",
        "revise_prediction_target_proposal",
        "confirm_prediction_target_setup",
        "get_prediction_target_status",
    }
)

EVIDENCE_SOURCES = frozenset(
    {
        "dataset_metadata",
        "uploaded_codebook",
        "uploaded_document",
        "user_statement",
        "semantic_inference",
        "unknown",
    }
)

POSITIVE_LABELS = {
    "case",
    "cancer",
    "disease",
    "diseased",
    "event",
    "malignant",
    "positive",
    "responder",
    "true",
    "yes",
}
NEGATIVE_LABELS = {
    "benign",
    "control",
    "healthy",
    "negative",
    "no",
    "non responder",
    "nonresponder",
    "false",
}


@dataclass
class PredictionTargetWorkflowState:
    """Structured proposal evidence and status for the first modeling stage."""

    status: str = "not_started"
    proposed_target_column: str | None = None
    target_candidate_reason: str | None = None
    target_candidate_confidence: str | None = None
    target_candidate_source: str = "unknown"
    target_values: list[Any] = field(default_factory=list)
    class_descriptions: dict[Any, str] = field(default_factory=dict)
    class_description_source: str = "unknown"
    proposed_positive_class: Any | None = None
    positive_class_reason: str | None = None
    positive_class_confidence: str | None = None
    positive_class_source: str = "unknown"
    candidate_columns: list[str] = field(default_factory=list)
    last_error: str | None = None

    def reset(self) -> None:
        self.__dict__.update(PredictionTargetWorkflowState().__dict__)


def create_prediction_target_proposal(
    state: Any,
    *,
    selected_target_col: str | None = None,
    target_selected_by_user: bool = False,
    explicit_positive_value: str = "",
    negative_class_description: str = "",
    positive_class_description: str = "",
    evidence_source: str = "unknown",
    evidence_reason: str = "",
) -> None:
    """Populate state.target_setup from data, metadata, or explicit evidence."""

    if state.df is None:
        raise ValueError("Attach a tabular dataset before defining the prediction target.")

    workflow = state.target_setup
    candidates = inspect_target_candidates(state.df)
    metadata_target = _metadata_target_column(state, list(state.df.columns))

    if selected_target_col is not None:
        target_col = selected_target_col
        candidate_reason = (
            evidence_reason.strip()
            or "The user selected this column as the outcome to predict."
        )
        candidate_confidence = "high"
        candidate_source = "user_statement" if target_selected_by_user else evidence_source
    elif metadata_target is not None:
        target_col = metadata_target
        candidate_reason = "Dataset metadata identifies this column as the target."
        candidate_confidence = "high"
        candidate_source = "dataset_metadata"
    else:
        plausible = _substantially_distinct_candidate(candidates)
        if plausible is None:
            workflow.reset()
            workflow.status = "awaiting_target_choice"
            workflow.candidate_columns = _ambiguous_candidate_names(candidates)
            workflow.last_error = None
            return
        target_col = str(plausible["column_name"])
        candidate_reason = str(plausible["reason"])
        candidate_confidence = _candidate_confidence(candidates)
        candidate_source = "semantic_inference"

    if target_col not in state.df.columns:
        raise ValueError(f"Column {target_col!r} is not present in the active dataset.")

    target_values = get_unique_non_null_target_values(state.df, target_col)
    workflow.reset()
    workflow.proposed_target_column = target_col
    workflow.target_candidate_reason = candidate_reason
    workflow.target_candidate_confidence = candidate_confidence
    workflow.target_candidate_source = _validated_source(candidate_source)
    workflow.target_values = list(target_values)
    workflow.candidate_columns = [str(item["column_name"]) for item in candidates]

    if len(target_values) != 2:
        workflow.status = "blocked_non_binary"
        workflow.last_error = (
            "The selected target has only one usable value and cannot be modeled."
            if len(target_values) == 1
            else "The selected target does not have exactly two usable values."
        )
        return

    descriptions, description_source = _metadata_class_descriptions(
        state,
        target_values,
    )
    proposed_positive, positive_reason, positive_confidence = _metadata_positive_class(
        state,
        target_values,
    )
    positive_source = "unknown"
    if proposed_positive is not None:
        configured_source = _validated_source(
            str((state.source_metadata or {}).get("positive_class_source", "dataset_metadata"))
        )
        positive_source = (
            configured_source if configured_source != "unknown" else "dataset_metadata"
        )

    supplied_descriptions = bool(
        negative_class_description.strip() or positive_class_description.strip()
    )
    if explicit_positive_value.strip():
        proposed_positive = match_target_value(explicit_positive_value, target_values)
        negative_value = _other_value(target_values, proposed_positive)
        if negative_class_description.strip():
            descriptions[negative_value] = negative_class_description.strip()
        if positive_class_description.strip():
            descriptions[proposed_positive] = positive_class_description.strip()
        description_source = _validated_source(evidence_source)
        positive_reason = evidence_reason.strip() or "The positive outcome was explicitly supplied."
        positive_confidence = "high"
        positive_source = _validated_source(evidence_source)
    elif supplied_descriptions:
        source = _validated_source(evidence_source)
        descriptions = _descriptions_in_observed_order(
            target_values,
            negative_class_description,
            positive_class_description,
            proposed_positive,
        )
        description_source = source

    if proposed_positive is None:
        inferred_positive = _infer_positive_from_labels(target_values, descriptions)
        if inferred_positive is not None:
            proposed_positive = inferred_positive
            positive_reason = "The class wording suggests a positive and negative outcome."
            positive_confidence = "medium"
            positive_source = (
                description_source
                if description_source != "unknown"
                else "semantic_inference"
            )
            if description_source == "unknown":
                description_source = "semantic_inference"
                descriptions = {
                    value: display_target_value(value) for value in target_values
                }

    workflow.class_descriptions = descriptions
    workflow.class_description_source = description_source
    workflow.proposed_positive_class = proposed_positive
    workflow.positive_class_reason = positive_reason
    workflow.positive_class_confidence = positive_confidence
    workflow.positive_class_source = positive_source
    workflow.status = (
        "awaiting_confirmation"
        if proposed_positive is not None
        else "awaiting_positive_class"
    )
    workflow.last_error = None


def compact_prediction_target_status(state: Any) -> dict[str, Any]:
    workflow = state.target_setup
    positive = workflow.proposed_positive_class
    negative = (
        _other_value(workflow.target_values, positive)
        if positive is not None and len(workflow.target_values) == 2
        else None
    )
    return {
        "workflow_stage": "prediction_target",
        "ok": workflow.status not in {"error"},
        "target_setup_status": workflow.status,
        "proposed_target_column": workflow.proposed_target_column,
        "target_candidate_reason": workflow.target_candidate_reason,
        "target_candidate_confidence": workflow.target_candidate_confidence,
        "target_candidate_source": workflow.target_candidate_source,
        "target_values": [display_target_value(value) for value in workflow.target_values],
        "class_descriptions": [
            {
                "value": display_target_value(value),
                "description": description,
            }
            for value, description in workflow.class_descriptions.items()
        ],
        "class_description_source": workflow.class_description_source,
        "proposed_positive_class": (
            display_target_value(positive) if positive is not None else None
        ),
        "proposed_negative_class": (
            display_target_value(negative) if negative is not None else None
        ),
        "positive_class_reason": workflow.positive_class_reason,
        "positive_class_confidence": workflow.positive_class_confidence,
        "positive_class_source": workflow.positive_class_source,
        "candidate_columns": workflow.candidate_columns,
        "last_error": workflow.last_error,
        "setup_complete": state.setup_status == "completed",
        "row_count": len(state.df) if state.df is not None else None,
        "feature_count": len(state.feature_names or []),
        "outcome_count": len(state.y) if state.y is not None else None,
        "target_mapping": _mapping_entries(state.target_mapping),
    }


def render_prediction_target_output(tool_name: str, data: Mapping[str, Any]) -> str:
    if data.get("ok") is False:
        return _render_error(data)
    if data.get("target_setup_status") == "complete":
        return _render_completion(data)
    return _render_proposal(data)


def _render_proposal(data: Mapping[str, Any]) -> str:
    status = data.get("target_setup_status")
    lines = ["## Let's define what the model should predict", ""]
    if status == "awaiting_target_choice":
        candidates = data.get("candidate_columns") or []
        if candidates:
            lines.extend(
                [
                    "Several columns could reasonably be the outcome:",
                    "",
                    *[f"- `{column}`" for column in candidates],
                    "",
                    "Which one should the model predict?",
                ]
            )
        else:
            lines.append("Which column should the model predict?")
        return "\n".join(lines)

    target = data.get("proposed_target_column")
    values = data.get("target_values") or []
    lines.extend(
        [
            f"`{target}` appears to be the most likely outcome column.",
            "",
            f"Your file contains {len(values)} distinct value{'s' if len(values) != 1 else ''} in this column:",
            "",
            *[f"- `{value}`" for value in values],
            "",
        ]
    )
    if status == "blocked_non_binary":
        value_count = len(values)
        if value_count == 1:
            explanation = (
                f"`{target}` contains only one distinct value, so there is no outcome "
                "difference for a model to learn. Please choose another outcome column."
            )
        else:
            explanation = (
                f"`{target}` contains {value_count} distinct values. The current modeling "
                "workflow supports two-outcome classification, so we need to choose "
                "another outcome column or define which two groups should be compared."
            )
        lines.extend(
            [
                explanation,
            ]
        )
        return "\n".join(lines)

    lines.append(
        "Because this column has two possible outcomes, this will be treated as a binary-classification problem."
    )
    positive = data.get("proposed_positive_class")
    negative = data.get("proposed_negative_class")
    descriptions = _description_lookup(data.get("class_descriptions"))
    source = data.get("class_description_source")
    if positive is None:
        displayed_values = " and ".join(f"`{value}`" for value in values)
        first_value, second_value = values
        lines.extend(
            [
                "",
                f"The file does not explain what {displayed_values} represent.",
                "",
                "**What does each value mean, and which one should be considered the positive outcome?**",
                "",
                "For example:",
                "",
                f"`{first_value}` means control, `{second_value}` means disease, and `{second_value}` should be positive.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "Here is the recommended setup:",
            "",
            f"- Outcome to predict: `{target}`",
            "- Prediction type: Binary classification",
            f"- Negative outcome: `{negative}`{_description_suffix(descriptions.get(str(negative)))}",
            f"- Positive outcome: `{positive}`{_description_suffix(descriptions.get(str(positive)))}",
            "",
            "Model encoding:",
            "",
            f"- `{negative}` -> `0.0`",
            f"- `{positive}` -> `1.0`",
            "",
            _source_sentence(str(source), str(data.get("positive_class_source"))),
            "",
            "Does this look right? Reply `Continue` to use it, or describe any correction in one message.",
        ]
    )
    return "\n".join(lines)


def _render_completion(data: Mapping[str, Any]) -> str:
    positive = data.get("proposed_positive_class")
    negative = data.get("proposed_negative_class")
    descriptions = _description_lookup(data.get("class_descriptions"))
    return "\n".join(
        [
            "## Prediction target confirmed",
            "",
            f"- **Outcome column:** `{data.get('proposed_target_column')}`",
            f"- **Negative outcome:** `{negative}`{_description_suffix(descriptions.get(str(negative)))}",
            f"- **Positive outcome:** `{positive}`{_description_suffix(descriptions.get(str(positive)))}",
            "",
            "### Model encoding",
            "",
            f"- `{negative}` -> `0.0`",
            f"- `{positive}` -> `1.0`",
            "",
            "The prediction target is now set.",
            "",
            f"- {data.get('row_count')} rows",
            f"- {data.get('feature_count')} input features",
            "",
            "No training/validation split, preprocessing, feature selection, or model training has run yet.",
            "",
            "The next stage will prepare the training and validation data.",
        ]
    )


def _render_error(data: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "## Let's define what the model should predict",
            "",
            str(data.get("message") or data.get("last_error") or "The target setup could not be updated."),
        ]
    )


def _substantially_distinct_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    top_score = int(candidates[0]["score"])
    second_score = int(candidates[1]["score"])
    return candidates[0] if top_score - second_score >= 20 else None


def _candidate_confidence(candidates: list[dict[str, object]]) -> str:
    if len(candidates) == 1:
        return "high" if int(candidates[0]["score"]) >= 100 else "medium"
    gap = int(candidates[0]["score"]) - int(candidates[1]["score"])
    return "high" if gap >= 50 else "medium"


def _ambiguous_candidate_names(candidates: list[dict[str, object]]) -> list[str]:
    if not candidates:
        return []
    top_score = int(candidates[0]["score"])
    return [
        str(candidate["column_name"])
        for candidate in candidates
        if top_score - int(candidate["score"]) < 20
    ]


def _metadata_target_column(state: Any, columns: list[str]) -> str | None:
    metadata = state.source_metadata or {}
    value = metadata.get("target_name") or metadata.get("target_col")
    return value if isinstance(value, str) and value in columns else None


def _metadata_class_descriptions(
    state: Any,
    target_values: list[Any],
) -> tuple[dict[Any, str], str]:
    metadata = state.source_metadata or {}
    raw = metadata.get("class_descriptions")
    if not isinstance(raw, Mapping):
        return {}, "unknown"
    descriptions: dict[Any, str] = {}
    for raw_value, description in raw.items():
        try:
            original_value = match_target_value(raw_value, target_values)
        except ValueError:
            continue
        if isinstance(description, str) and description.strip():
            descriptions[original_value] = description.strip()
    if not descriptions:
        return {}, "unknown"
    configured_source = _validated_source(
        str(metadata.get("class_description_source", "dataset_metadata"))
    )
    return descriptions, (
        configured_source if configured_source != "unknown" else "dataset_metadata"
    )


def _metadata_positive_class(
    state: Any,
    target_values: list[Any],
) -> tuple[Any | None, str | None, str | None]:
    metadata = state.source_metadata or {}
    explicit = metadata.get("positive_class_value")
    if explicit is not None:
        try:
            value = match_target_value(explicit, target_values)
        except ValueError:
            pass
        else:
            return value, "Dataset metadata explicitly identifies the positive outcome.", "high"
    mapping = metadata.get("target_mapping")
    if isinstance(mapping, Mapping):
        for raw_value, encoded_value in mapping.items():
            if encoded_value in {1, 1.0}:
                try:
                    value = match_target_value(raw_value, target_values)
                except ValueError:
                    continue
                return value, "Dataset metadata explicitly supplies the target mapping.", "high"
    return None, None, None


def _infer_positive_from_labels(
    target_values: list[Any],
    descriptions: Mapping[Any, str],
) -> Any | None:
    positive_matches: list[Any] = []
    negative_matches: list[Any] = []
    for value in target_values:
        text = descriptions.get(value, display_target_value(value))
        polarity = _meaning_polarity(text)
        if polarity == "positive":
            positive_matches.append(value)
        if polarity == "negative":
            negative_matches.append(value)
    if len(positive_matches) == 1 and len(negative_matches) == 1:
        return positive_matches[0]
    return None


def _descriptions_in_observed_order(
    target_values: list[Any],
    negative_description: str,
    positive_description: str,
    proposed_positive: Any | None,
) -> dict[Any, str]:
    if proposed_positive is None:
        return {}
    negative = _other_value(target_values, proposed_positive)
    descriptions: dict[Any, str] = {}
    if negative_description.strip():
        descriptions[negative] = negative_description.strip()
    if positive_description.strip():
        descriptions[proposed_positive] = positive_description.strip()
    return descriptions


def _other_value(values: list[Any], selected: Any) -> Any:
    others = [value for value in values if not _values_equal(value, selected)]
    if len(others) != 1:
        raise ValueError("The negative outcome could not be determined unambiguously.")
    return others[0]


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _validated_source(source: str) -> str:
    normalized = source.strip().casefold().replace(" ", "_")
    return normalized if normalized in EVIDENCE_SOURCES else "unknown"


def _normalize_meaning(value: str) -> str:
    return " ".join(
        value.strip().casefold().replace("_", " ").replace("-", " ").split()
    )


def _mapping_entries(mapping: Mapping[Any, float] | None) -> list[dict[str, Any]]:
    return [
        {
            "original_value": display_target_value(value),
            "encoded_value": float(encoded),
        }
        for value, encoded in (mapping or {}).items()
    ]


def _description_lookup(entries: Any) -> dict[str, str]:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return {}
    return {
        str(entry.get("value")): str(entry.get("description"))
        for entry in entries
        if isinstance(entry, Mapping)
    }


def _description_suffix(description: str | None) -> str:
    return f" - {description}" if description else ""


def _source_sentence(description_source: str, positive_source: str) -> str:
    description_message = {
        "dataset_metadata": "These meanings come from information provided with the dataset.",
        "uploaded_codebook": "These meanings come from the uploaded codebook.",
        "uploaded_document": "These meanings come from an uploaded document.",
        "user_statement": "These meanings come from information you supplied.",
        "semantic_inference": "The labels suggest this interpretation, but please confirm it.",
    }.get(description_source)
    if description_message:
        return description_message
    return {
        "dataset_metadata": "The positive outcome is identified by information provided with the dataset.",
        "uploaded_codebook": "The positive outcome is identified by the uploaded codebook.",
        "uploaded_document": "The positive outcome is identified by an uploaded document.",
        "user_statement": "The positive outcome follows the information you supplied.",
        "semantic_inference": "The labels suggest this positive outcome, but please confirm it.",
    }.get(
        positive_source,
        "The value meanings are not documented, so please confirm this mapping.",
    )


def _meaning_polarity(value: str) -> str | None:
    normalized = _normalize_meaning(value)
    if normalized in NEGATIVE_LABELS:
        return "negative"
    if normalized in POSITIVE_LABELS:
        return "positive"
    tokens = set(normalized.split())
    if tokens.intersection(NEGATIVE_LABELS):
        return "negative"
    if tokens.intersection(POSITIVE_LABELS):
        return "positive"
    return None
