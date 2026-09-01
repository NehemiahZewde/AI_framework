"""Chainlit entrypoint for the general chat with OpenAI-native attachments."""

from __future__ import annotations

import traceback

import chainlit as cl

from agent import ConversationalAgent, test_openai_api_key
from attachment_manager import AttachmentMetadata, attachment_manager
from openai_file_service import AttachmentUploadError, upload_attachment
from session_manager import conversation_sessions
from tabular_loader import load_tabular_workspace
from tabular_workspace import MLAgentContext, TabularWorkspace, tabular_workspaces


OPENAI_API_KEY_SESSION_KEY = "openai_api_key"


@cl.on_chat_start
async def on_chat_start() -> None:
    """Validate the UI-provided API key and prepare a new chat session."""

    api_key = await _ask_for_valid_api_key()
    if api_key is None:
        return

    try:
        conversation_sessions.get_session(_get_chainlit_session_id())
    except Exception as exc:
        _log_sanitized_exception("Conversation session initialization failed", exc, api_key)
        await cl.Message(
            content="I could not start the conversation session. Please start a new chat."
        ).send()
        return

    await cl.Message(content="API key validated. What would you like to talk about?").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Upload any attachments, then run one continuous conversational turn."""

    api_key = cl.user_session.get(OPENAI_API_KEY_SESSION_KEY)
    if not api_key:
        await cl.Message(content="Start a new chat and enter an API key first.").send()
        return

    try:
        chainlit_session_id = _get_chainlit_session_id()
    except Exception as exc:
        _log_sanitized_exception("Conversation session lookup failed", exc, api_key)
        await cl.Message(
            content="I could not access the conversation session. Please start a new chat."
        ).send()
        return

    attachments = _get_file_attachments(message)
    uploaded_attachments = await _upload_message_attachments(
        attachments,
        api_key,
        chainlit_session_id,
    )
    local_status_notes = _prepare_local_tabular_workspaces(
        attachments,
        chainlit_session_id,
        {attachment.original_filename for attachment in uploaded_attachments},
    )
    user_message = (message.content or "").strip()

    if attachments and not user_message and not uploaded_attachments:
        if local_status_notes:
            await cl.Message(content="\n\n".join(local_status_notes)).send()
        return
    if not user_message:
        if uploaded_attachments:
            user_message = _attachment_only_instruction(uploaded_attachments)
        else:
            await cl.Message(content="Please enter a message or attach a supported file.").send()
            return

    try:
        session = conversation_sessions.get_session(chainlit_session_id)
        agent = ConversationalAgent(api_key=api_key)
        response = await agent.respond(
            user_message,
            session=session,
            file_ids=[attachment.openai_file_id for attachment in uploaded_attachments],
            runtime_context=MLAgentContext(
                tabular_workspace=tabular_workspaces.get(chainlit_session_id)
            ),
        )
    except Exception as exc:
        _log_sanitized_exception("Conversational agent request failed", exc, api_key)
        if local_status_notes:
            await cl.Message(content="\n\n".join(local_status_notes)).send()
        await cl.Message(content="I could not complete that response. Please try again.").send()
        return

    if local_status_notes:
        response = "\n\n".join([*local_status_notes, response])
    await cl.Message(content=response).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Release this chat's temporary conversation and attachment metadata."""

    try:
        chainlit_session_id = _get_chainlit_session_id()
        conversation_sessions.remove_session(chainlit_session_id)
        attachment_manager.remove_session(chainlit_session_id)
        tabular_workspaces.remove_session(chainlit_session_id)
    except Exception:
        # A session may have ended before it was initialized.
        return


async def _ask_for_valid_api_key() -> str | None:
    """Request, validate, and retain the API key only for this Chainlit chat."""

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
            return api_key

        await cl.Message(
            content="That API key could not be validated. Please check the key and try again."
        ).send()

    await cl.Message(content="API key was not validated. Start a new chat to try again.").send()
    return None


def _get_file_attachments(message: cl.Message) -> list[object]:
    """Return Chainlit elements that have an attachment name or mounted path."""

    return [
        element
        for element in message.elements or []
        if getattr(element, "name", None) or getattr(element, "path", None)
    ]


async def _upload_message_attachments(
    attachments: list[object],
    api_key: str,
    chainlit_session_id: str,
) -> list[AttachmentMetadata]:
    """Upload each attachment once and retain only its OpenAI metadata."""

    uploaded_attachments: list[AttachmentMetadata] = []
    for attachment in attachments:
        file_name = getattr(attachment, "name", None)
        file_path = getattr(attachment, "path", None)
        if not isinstance(file_name, str) or not file_name.strip():
            await cl.Message(content="The attachment does not have a usable filename.").send()
            continue
        if not file_path:
            await cl.Message(
                content="The attached file could not be accessed. The general chat is still available."
            ).send()
            continue

        raw_mime_type = getattr(attachment, "mime", None) or getattr(
            attachment, "content_type", None
        )
        mime_type = str(raw_mime_type) if raw_mime_type else None
        try:
            uploaded_attachments.append(
                await upload_attachment(
                    file_path=file_path,
                    original_filename=file_name,
                    mime_type=mime_type,
                    api_key=api_key,
                )
            )
        except AttachmentUploadError as exc:
            await cl.Message(content=str(exc)).send()

    attachment_manager.add_many(chainlit_session_id, uploaded_attachments)
    return uploaded_attachments


def _prepare_local_tabular_workspaces(
    attachments: list[object],
    chainlit_session_id: str,
    native_uploaded_file_names: set[str],
) -> list[str]:
    """Prepare supported tables locally without changing native-file behavior."""

    status_notes: list[str] = []
    current_workspace = tabular_workspaces.get(chainlit_session_id)

    for attachment in attachments:
        file_name = getattr(attachment, "name", None)
        file_path = getattr(attachment, "path", None)
        if not isinstance(file_name, str) or not file_name.strip() or not file_path:
            continue

        raw_mime_type = getattr(attachment, "mime", None) or getattr(
            attachment, "content_type", None
        )
        mime_type = str(raw_mime_type) if raw_mime_type else None
        result = load_tabular_workspace(
            file_path=file_path,
            original_file_name=file_name,
            content_type=mime_type,
        )
        if not result.is_supported_tabular_file:
            continue
        if result.workspace is not None:
            replaced_existing = current_workspace is not None
            current_workspace = result.workspace
            tabular_workspaces.set(chainlit_session_id, result.workspace)
            status_notes.append(
                _render_local_workspace_success(result.workspace, replaced_existing)
            )
            continue

        conversation_note = (
            "The file remains available for normal OpenAI-native conversation."
            if file_name in native_uploaded_file_names
            else "The general chat remains available."
        )
        status_notes.append(
            f"`{file_name}` could not be prepared as a local pandas table: "
            f"{result.local_load_error} {conversation_note}"
        )

    return status_notes


def _render_local_workspace_success(
    workspace: TabularWorkspace,
    replaced_existing: bool,
) -> str:
    """Render one compact local-workspace status note for the agent response."""

    lines = [
        f"`{workspace.original_file_name}` is also prepared locally for future ML-framework operations.",
        f"- Rows: {workspace.row_count}",
        f"- Columns: {workspace.column_count}",
    ]
    if workspace.active_sheet_name:
        lines.append(f"- Active sheet: {workspace.active_sheet_name}")
        lines.append(f"- Available sheets: {', '.join(workspace.sheet_names)}")
        lines.append("- Only the active sheet is stored as the local pandas DataFrame.")
    if replaced_existing:
        lines.append("- The previous active local table was replaced.")
    return "\n".join(lines)


def _attachment_only_instruction(attachments: list[AttachmentMetadata]) -> str:
    file_names = ", ".join(f"`{attachment.original_filename}`" for attachment in attachments)
    return (
        f"The user attached {file_names}. Briefly acknowledge the file and ask what "
        "they would like to know about it. Do not perform a long analysis unless requested."
    )


def _get_chainlit_session_id() -> str:
    """Return Chainlit's stable identifier for the current chat."""

    session_id = cl.user_session.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("Chainlit did not provide a session identifier.")
    return session_id


def _log_sanitized_exception(label: str, exc: Exception, api_key: str | None) -> None:
    """Write diagnostics without allowing the session API key into the terminal."""

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
