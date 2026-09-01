"""Chainlit entrypoint for the Dataset Explorer Agent."""

from __future__ import annotations

import traceback

import chainlit as cl

from agent import DatasetExplorerAgent, test_openai_api_key
from dataset_profile import (
    build_draft_ml_setup,
    get_top_target_column,
    is_affirmative,
    load_csv_dataframe,
    parse_yes_no,
    profile_dataframe,
    render_dataset_profile_markdown,
    render_draft_setup_markdown,
    resolve_column_name,
)
from dataset_setup import (
    build_standardized_dataset_setup,
    format_target_values,
    get_unique_non_null_target_values,
    infer_feature_groups_for_preprocessing,
    match_target_value,
    render_feature_group_review,
    render_part1_setup_review,
)
from state import (
    CSV_FILE_NAME_KEY,
    CONFIRMED_FEATURE_GROUPS_KEY,
    DATASET_PROFILE_KEY,
    DRAFT_SETUP_KEY,
    FEATURE_GROUPS_KEY,
    DatasetExplorerState,
    OPENAI_API_KEY_SESSION_KEY,
    STANDARDIZED_DATASET_SETUP_KEY,
)


BINARY_QUESTION = "Is this a binary classification task? Reply yes or no."


@cl.on_chat_start
async def on_chat_start() -> None:
    """Start the one-file dataset exploration flow."""

    api_key = await _ask_for_valid_api_key()
    if api_key is None:
        return

    await cl.Message(
        content=(
            "Upload one CSV file. I will profile it, ask you to confirm the "
            "target, and draft the ML setup. No training or feature selection "
            "will run."
        )
    ).send()

    files = await _ask_for_one_csv()
    if not files:
        await cl.Message(content="Please attach a CSV file.").send()
        return

    await _process_csv_upload(files[0], api_key)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Process paperclip CSV uploads or keep follow-up behavior narrow."""

    profile = cl.user_session.get(DATASET_PROFILE_KEY)
    if profile is not None:
        await cl.Message(
            content=(
                "Version 1 only profiles the uploaded CSV and drafts the setup. "
                "Start a new chat to explore another dataset."
            )
        ).send()
        return

    api_key = cl.user_session.get(OPENAI_API_KEY_SESSION_KEY)
    if not api_key:
        await cl.Message(content="Start a new chat and enter an API key first.").send()
        return

    csv_file = _get_csv_attachment(message)
    if csv_file is None:
        await cl.Message(content="Please attach a CSV file.").send()
        return

    await _process_csv_upload(csv_file, api_key)


async def _process_csv_upload(csv_file: object, api_key: str) -> None:
    if not _is_csv_file(csv_file):
        await cl.Message(content="Please attach a CSV file.").send()
        return

    csv_path = getattr(csv_file, "path", None)
    if not csv_path:
        await cl.Message(content="Could not read the attached CSV file.").send()
        return

    csv_name = getattr(csv_file, "name", "uploaded.csv")
    explorer = DatasetExplorerAgent(api_key=api_key)

    try:
        df = load_csv_dataframe(csv_path)
        profile = profile_dataframe(df, source_name=csv_name)
    except Exception as exc:
        await cl.Message(content=f"Could not load the CSV: {exc}").send()
        return

    state = DatasetExplorerState(
        csv_file_name=csv_name,
        openai_api_key=api_key,
        profile=profile,
    )
    cl.user_session.set(CSV_FILE_NAME_KEY, state.csv_file_name)
    cl.user_session.set(OPENAI_API_KEY_SESSION_KEY, state.openai_api_key)
    cl.user_session.set(DATASET_PROFILE_KEY, state.profile)

    try:
        profile_report = await explorer.profile_report(csv_path)
    except Exception as exc:
        _log_sanitized_exception("Dataset profile agent call failed", exc, api_key)
        profile_report = (
            f"{render_dataset_profile_markdown(profile)}\n\n"
            "_Agent response failed after API key validation, so the deterministic "
            "pandas profile is shown instead._"
        )

    await cl.Message(content=profile_report).send()

    target_column = await _ask_for_target(profile)
    if target_column is None:
        return

    is_binary = await _ask_for_binary_task(profile, target_column)
    if is_binary is None:
        return

    setup = build_draft_ml_setup(
        profile=profile,
        target_column=target_column,
        is_binary_classification=is_binary,
    )
    cl.user_session.set(DRAFT_SETUP_KEY, setup)

    if not is_binary:
        await cl.Message(
            content=(
                f"{render_draft_setup_markdown(setup)}\n\n"
                "Part 1 standardized setup was not created because binary "
                "classification was not confirmed."
            )
        ).send()
        return

    target_values = await _get_valid_binary_target_values(df, target_column)
    if target_values is None:
        return

    positive_class_value = await _ask_for_positive_class_value(target_values)
    if positive_class_value is None:
        return

    try:
        standardized_setup = build_standardized_dataset_setup(
            df=df,
            target_col=target_column,
            positive_class_value=positive_class_value,
        )
    except Exception as exc:
        await cl.Message(content=f"Could not create Part 1 setup: {exc}").send()
        return

    cl.user_session.set(STANDARDIZED_DATASET_SETUP_KEY, standardized_setup)
    await cl.Message(content=render_part1_setup_review(standardized_setup)).send()

    feature_groups = infer_feature_groups_for_preprocessing(standardized_setup["X"])
    cl.user_session.set(FEATURE_GROUPS_KEY, feature_groups)

    if await _ask_to_confirm_feature_groups(feature_groups):
        cl.user_session.set(CONFIRMED_FEATURE_GROUPS_KEY, feature_groups)
        await cl.Message(content="Feature-group setup confirmed.").send()


async def _ask_for_one_csv() -> list | None:
    return await cl.AskFileMessage(
        content="Upload one CSV file.",
        accept={"text/csv": [".csv"], "application/vnd.ms-excel": [".csv"]},
        max_files=1,
        max_size_mb=200,
        timeout=600,
    ).send()


def _get_csv_attachment(message: cl.Message) -> object | None:
    for element in message.elements or []:
        if _is_csv_file(element):
            return element
    return None


def _is_csv_file(file_obj: object) -> bool:
    name = str(getattr(file_obj, "name", "") or "")
    path = str(getattr(file_obj, "path", "") or "")
    return name.lower().endswith(".csv") or path.lower().endswith(".csv")


async def _get_valid_binary_target_values(df: object, target_column: str) -> list | None:
    try:
        target_values = get_unique_non_null_target_values(df, target_column)
    except Exception as exc:
        await cl.Message(content=f"Could not inspect target values: {exc}").send()
        return None

    if len(target_values) != 2:
        await cl.Message(
            content=(
                f"Part 1 setup currently requires exactly two non-null target "
                f"values. `{target_column}` has {len(target_values)}."
            )
        ).send()
        return None

    return target_values


async def _ask_for_positive_class_value(target_values: list) -> object | None:
    values_text = format_target_values(target_values)
    prompt = (
        "Which target value should be treated as the positive class?\n\n"
        f"Available values: {values_text}"
    )

    for _ in range(3):
        response = await cl.AskUserMessage(content=prompt, timeout=600).send()
        if response is None:
            await cl.Message(content="No positive class value was confirmed.").send()
            return None

        matched_value = match_target_value(response["output"], target_values)
        if matched_value is not None:
            await cl.Message(
                content=f"Positive class confirmed: `{format_target_values([matched_value])}`"
            ).send()
            return matched_value

        await cl.Message(
            content=(
                "That value did not match the available target values. "
                f"Choose one of: {values_text}"
            )
        ).send()

    await cl.Message(content="Positive class value was not confirmed.").send()
    return None


async def _ask_to_confirm_feature_groups(feature_groups: dict) -> bool:
    review = render_feature_group_review(feature_groups)

    for _ in range(3):
        response = await cl.AskUserMessage(content=review, timeout=600).send()
        if response is None:
            await cl.Message(content="Feature-group setup was not confirmed.").send()
            return False

        if is_affirmative(response["output"]):
            return True

        await cl.Message(
            content=(
                "Manual feature-group editing will be added next. For now, "
                "please reply yes if this setup is acceptable."
            )
        ).send()

    await cl.Message(content="Feature-group setup was not confirmed.").send()
    return False


async def _ask_for_valid_api_key() -> str | None:
    for _ in range(3):
        response = await cl.AskUserMessage(
            content=(
                "Paste your OpenAI API key to start. It will only be kept in "
                "this Chainlit session and will not be saved to disk. Do not "
                "share screenshots while the key is visible."
            ),
            timeout=600,
        ).send()
        if response is None:
            await cl.Message(content="No API key was provided.").send()
            return None

        api_key = response["output"].strip()
        await cl.Message(content="Testing the API key...").send()

        if await test_openai_api_key(api_key):
            cl.user_session.set(OPENAI_API_KEY_SESSION_KEY, api_key)
            await cl.Message(content="API key validated.").send()
            return api_key

        await cl.Message(
            content=(
                "That API key could not be validated. Please check the key and "
                "try again."
            )
        ).send()

    await cl.Message(content="API key was not validated. Start a new chat to try again.").send()
    return None


def _log_sanitized_exception(label: str, exc: Exception, api_key: str | None) -> None:
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"{label}:\n{_redact_sensitive_text(traceback_text, api_key)}")


def _redact_sensitive_text(text: str, api_key: str | None) -> str:
    redacted = text
    if api_key:
        redacted = redacted.replace(api_key, "[REDACTED_OPENAI_API_KEY]")

    for token in ("sk-", "sess-"):
        redacted = _redact_token_prefix(redacted, token)

    return redacted


def _redact_token_prefix(text: str, prefix: str) -> str:
    pieces = text.split(prefix)
    if len(pieces) == 1:
        return text

    redacted = [pieces[0]]
    for piece in pieces[1:]:
        suffix = []
        remainder_index = 0
        for index, character in enumerate(piece):
            if character.isspace() or character in {'"', "'", "`", ")", "]", "}"}:
                remainder_index = index
                break
            suffix.append(character)
        else:
            remainder_index = len(piece)

        if len(suffix) >= 8:
            redacted.append("[REDACTED_OPENAI_API_KEY]")
            redacted.append(piece[remainder_index:])
        else:
            redacted.append(prefix)
            redacted.append(piece)

    return "".join(redacted)


async def _ask_for_target(profile: dict) -> str | None:
    suggested_target = get_top_target_column(profile)
    if suggested_target:
        prompt = (
            f"I think `{suggested_target}` is the target column. Reply `yes` "
            "to confirm, or type a different column name."
        )
    else:
        prompt = "I could not identify a likely target column. Type the target column name."

    for _ in range(3):
        response = await cl.AskUserMessage(content=prompt, timeout=600).send()
        if response is None:
            await cl.Message(content="No target column was confirmed.").send()
            return None

        user_value = response["output"]
        if suggested_target and is_affirmative(user_value):
            await cl.Message(content=f"Target column confirmed: `{suggested_target}`").send()
            return suggested_target

        try:
            target_column = resolve_column_name(profile, user_value)
        except ValueError as exc:
            await cl.Message(content=str(exc)).send()
            continue

        await cl.Message(content=f"Target column confirmed: `{target_column}`").send()
        return target_column

    await cl.Message(content="Target column was not confirmed.").send()
    return None


async def _ask_for_binary_task(profile: dict, target_column: str) -> bool | None:
    unique_count = profile["unique_counts_by_column"].get(target_column)
    prompt = f"{BINARY_QUESTION} `{target_column}` has {unique_count} unique value(s)."

    for _ in range(3):
        response = await cl.AskUserMessage(content=prompt, timeout=600).send()
        if response is None:
            await cl.Message(content="Binary classification status was not confirmed.").send()
            return None

        parsed = parse_yes_no(response["output"])
        if parsed is None:
            await cl.Message(content="Please reply with yes or no.").send()
            continue

        task_label = "binary classification" if parsed else "not binary classification"
        await cl.Message(content=f"Task type confirmed: {task_label}.").send()
        return parsed

    await cl.Message(content="Binary classification status was not confirmed.").send()
    return None
