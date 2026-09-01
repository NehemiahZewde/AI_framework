"""Chainlit entrypoint for the Phase 1 conversational chat agent."""

from __future__ import annotations

import traceback

import chainlit as cl

from agent import ConversationalAgent, test_openai_api_key
from session_manager import conversation_sessions


OPENAI_API_KEY_SESSION_KEY = "openai_api_key"


@cl.on_chat_start
async def on_chat_start() -> None:
    api_key = await _ask_for_valid_api_key()
    if api_key is None:
        return
    try:
        conversation_sessions.get_session(_get_chainlit_session_id())
    except Exception as exc:
        _log_sanitized_exception("Conversation session initialization failed", exc, api_key)
        await cl.Message(content="I could not start the conversation session. Please start a new chat.").send()
        return
    await cl.Message(content="API key validated. What would you like to talk about?").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    api_key = cl.user_session.get(OPENAI_API_KEY_SESSION_KEY)
    if not api_key:
        await cl.Message(content="Start a new chat and enter an API key first.").send()
        return
    user_message = (message.content or "").strip()
    if not user_message:
        await cl.Message(content="Please enter a message.").send()
        return
    try:
        session = conversation_sessions.get_session(_get_chainlit_session_id())
        response = await ConversationalAgent(api_key=api_key).respond(user_message, session=session)
    except Exception as exc:
        _log_sanitized_exception("Conversational agent request failed", exc, api_key)
        await cl.Message(content="I could not complete that response. Please try again.").send()
        return
    await cl.Message(content=response).send()


@cl.on_chat_end
async def on_chat_end() -> None:
    try:
        conversation_sessions.remove_session(_get_chainlit_session_id())
    except Exception:
        return


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
            return api_key
        await cl.Message(content="That API key could not be validated. Please check the key and try again.").send()
    await cl.Message(content="API key was not validated. Start a new chat to try again.").send()
    return None


def _get_chainlit_session_id() -> str:
    session_id = cl.user_session.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise RuntimeError("Chainlit did not provide a session identifier.")
    return session_id


def _log_sanitized_exception(label: str, exc: Exception, api_key: str | None) -> None:
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"{label}:\n{_redact_sensitive_text(traceback_text, api_key)}")


def _redact_sensitive_text(text: str, api_key: str | None) -> str:
    redacted = text.replace(api_key, "[REDACTED_OPENAI_API_KEY]") if api_key else text
    for token in ("sk-", "sess-"):
        redacted = _redact_token_prefix(redacted, token)
    return redacted


def _redact_token_prefix(text: str, prefix: str) -> str:
    pieces = text.split(prefix)
    if len(pieces) == 1:
        return text
    redacted = [pieces[0]]
    for piece in pieces[1:]:
        remainder_index = next(
            (index for index, character in enumerate(piece) if character.isspace() or character in {'"', "'", "`", ")", "]", "}"}),
            len(piece),
        )
        if remainder_index >= 8:
            redacted.extend(["[REDACTED_OPENAI_API_KEY]", piece[remainder_index:]])
        else:
            redacted.extend([prefix, piece])
    return "".join(redacted)
