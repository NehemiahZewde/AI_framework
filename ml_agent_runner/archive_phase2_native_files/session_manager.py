"""In-memory mapping between Chainlit chats and Agents SDK sessions."""

from __future__ import annotations

from agents import SQLiteSession


class ConversationSessionManager:
    """Keep one temporary Agents SDK session per active Chainlit chat."""

    def __init__(self) -> None:
        self._sessions: dict[str, SQLiteSession] = {}

    def get_session(self, chainlit_session_id: str) -> SQLiteSession:
        """Return the session for a Chainlit chat, creating it when needed."""

        if not chainlit_session_id:
            raise ValueError("A Chainlit session identifier is required.")

        session = self._sessions.get(chainlit_session_id)
        if session is None:
            session = SQLiteSession(chainlit_session_id, ":memory:")
            self._sessions[chainlit_session_id] = session
        return session

    def remove_session(self, chainlit_session_id: str) -> None:
        """Forget a completed chat's temporary in-memory session."""

        self._sessions.pop(chainlit_session_id, None)


conversation_sessions = ConversationSessionManager()
